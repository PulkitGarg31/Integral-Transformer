"""Measure wall-clock training time per epoch for every model in the roster.

The training loop records losses and F1 but never recorded time, so the paper
has no per-epoch cost numbers. This script produces them without retraining:
it replays the EXACT train step (fp16 autocast + GradScaler + clip + AdamW +
scheduler) and the EXACT validation pass from src/training.py on a sample of
batches, then extrapolates to a full epoch using the real loader lengths.

Run it on the same GPU the paper numbers came from, with the same batch size
and the same verdict-hidden regime:

    python benchmark_epoch_time.py --sc-path /workspace/pdfs/Supreme_Court

Writes benchmark_epoch_time.json (and prints a table) with, per model:
  train_s_per_batch, val_s_per_batch, train_epoch_s, val_epoch_s, epoch_s.

--full-epoch times one genuine epoch per model instead of extrapolating
(accurate but ~as slow as a real training run).
"""

import argparse
import gc
import json
import os
import sys
import time
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sc-path', default=None, help='Supreme Court PDF root')
    p.add_argument('--delhi-path', default=None, help='unused; accepted for parity')
    p.add_argument('--train-mode', default='hidden', choices=['hidden', 'full'],
                   help='hidden = verdict-hidden regime used for the paper tables')
    p.add_argument('--n-remove', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=None,
                   help='defaults to Config.batch_size (28)')
    p.add_argument('--seed', type=int, default=16911)
    p.add_argument('--random-seed', action='store_true')
    # build_config() in main.py reads these three off the namespace; they are
    # irrelevant to timing but must exist or it raises AttributeError.
    p.add_argument('--epochs', type=int, default=None,
                   help='only feeds the LR scheduler length, not the batch count')
    p.add_argument('--resume', action='store_true',
                   help='accepted for parity; the benchmark never loads checkpoints')
    p.add_argument('--warmup', type=int, default=5,
                   help='untimed batches first (cuDNN autotune + CUDA graphs warm)')
    p.add_argument('--steps', type=int, default=40,
                   help='timed train batches per model')
    p.add_argument('--val-steps', type=int, default=20,
                   help='timed val batches per model')
    p.add_argument('--full-epoch', action='store_true',
                   help='time one real epoch per model instead of extrapolating')
    p.add_argument('--repeats', type=int, default=3,
                   help='timing passes per model; the median is reported and the '
                        'min/max spread is recorded so noise is visible')
    p.add_argument('--models', default=None,
                   help='comma-separated subset, e.g. integral,baseline')
    p.add_argument('--out', default='benchmark_epoch_time.json')
    return p.parse_args()


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_train(model, loader, config, model_name, warmup, steps, torch, seed):
    """Replay src/training.py's inner train step and return seconds/batch.

    `seed` re-seeds the global RNG immediately before iterating, so the
    shuffling RandomSampler draws the SAME permutation for every model. Without
    it each model times a different 40 documents, and since encode_chunks only
    encodes real chunks, a luckier draw of short judgments looks like a faster
    model — a few percent of pure sampling artifact.
    """
    from torch.amp import autocast, GradScaler
    from transformers import get_linear_schedule_with_warmup
    from src.training import build_optimizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).train()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(model, model_name, config)
    accum = config.gradient_accumulation_steps
    total_steps = max(len(loader) // accum, 1) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps)
    scaler = GradScaler('cuda', enabled=config.use_fp16)

    optimizer.zero_grad()
    timed, elapsed, t0 = 0, 0.0, None
    torch.manual_seed(seed)          # identical batch order for every model
    for step, batch in enumerate(loader):
        if step == warmup:
            _sync(torch)
            t0 = time.perf_counter()
        ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        cmask = batch['chunk_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        with autocast('cuda', enabled=config.use_fp16):
            logits = model(ids, mask, cmask)
            loss = criterion(logits, labels) / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        if t0 is not None:
            timed += 1
            if steps is not None and timed >= steps:
                break
    _sync(torch)
    elapsed = time.perf_counter() - t0
    del optimizer, scheduler, scaler
    return elapsed / max(timed, 1), timed


def time_val(model, loader, config, warmup, steps, torch):
    """Replay src/evaluation.py's eval pass and return seconds/batch."""
    from torch.amp import autocast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    timed, t0 = 0, None
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step == warmup:
                _sync(torch)
                t0 = time.perf_counter()
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            cmask = batch['chunk_mask'].to(device, non_blocking=True)
            with autocast('cuda', enabled=config.use_fp16):
                model(ids, mask, cmask)
            if t0 is not None:
                timed += 1
                if steps is not None and timed >= steps:
                    break
    _sync(torch)
    return (time.perf_counter() - t0) / max(timed, 1), timed


def main():
    args = parse_args()
    import torch
    import main as pipeline
    from src.dataset import ChunkRemovedDataset, make_dataloader
    from src.models import MODEL_NAMES, MODEL_REGISTRY

    config = pipeline.build_config(args)
    train_ds, val_ds, _ = pipeline.load_or_build_sc_datasets(config, args)
    if args.train_mode == 'hidden':
        print(f"🔪 Verdict-hidden: removing last {args.n_remove} chunk(s)")
        train_ds = ChunkRemovedDataset(train_ds, args.n_remove)
        val_ds = ChunkRemovedDataset(val_ds, args.n_remove)
    train_loader = make_dataloader(train_ds, config.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, config.batch_size, shuffle=False)

    names = [n.strip() for n in args.models.split(',')] if args.models else list(MODEL_NAMES)
    warmup = 0 if args.full_epoch else args.warmup
    steps = None if args.full_epoch else args.steps
    val_steps = None if args.full_epoch else args.val_steps

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {gpu} | batch={config.batch_size} | fp16={config.use_fp16}")
    print(f"train batches/epoch={len(train_loader)}  val batches/epoch={len(val_loader)}")

    results = {}
    for name in names:
        print(f"\n▶ Timing {name.upper()} ...")
        model = MODEL_REGISTRY[name](config)
        params = sum(p.numel() for p in model.parameters())
        tr_runs, va_runs, tr_n, va_n = [], [], 0, 0
        for rep in range(max(args.repeats, 1)):
            s, tr_n = time_train(model, train_loader, config, name, warmup,
                                 steps, torch, args.seed)
            tr_runs.append(s)
            s, va_n = time_val(model, val_loader, config, warmup, val_steps, torch)
            va_runs.append(s)
            print(f"  pass {rep+1}/{max(args.repeats, 1)}: "
                  f"{tr_runs[-1]*1000:.1f} ms/train batch, {va_runs[-1]*1000:.1f} ms/val batch")
        tr_s, va_s = median(tr_runs), median(va_runs)
        train_epoch = tr_s * len(train_loader)
        val_epoch = va_s * len(val_loader)
        # spread over the whole epoch, so the plot's error bar is in the same
        # units as the bar it sits on
        epoch_runs = sorted(t * len(train_loader) + v * len(val_loader)
                            for t, v in zip(tr_runs, va_runs))
        results[name] = {
            'params': params,
            'train_s_per_batch': tr_s,
            'val_s_per_batch': va_s,
            'train_epoch_s': train_epoch,
            'val_epoch_s': val_epoch,
            'epoch_s': train_epoch + val_epoch,
            'epoch_s_runs': epoch_runs,
            'epoch_s_min': epoch_runs[0],
            'epoch_s_max': epoch_runs[-1],
            'repeats': len(epoch_runs),
            'timed_train_batches': tr_n,
            'timed_val_batches': va_n,
        }
        spread = (epoch_runs[-1] - epoch_runs[0]) / epoch_runs[0] * 100
        print(f"  median → {(train_epoch + val_epoch)/60:.2f} min/epoch "
              f"(spread {spread:.1f}% over {len(epoch_runs)} pass(es))")
        del model
        torch.cuda.empty_cache()
        gc.collect()

    payload = {
        'device': gpu,
        'batch_size': config.batch_size,
        'fp16': config.use_fp16,
        'max_chunks': config.max_chunks,
        'max_length': config.max_length,
        'train_mode': args.train_mode,
        'n_remove': args.n_remove,
        'seed': args.seed,
        'train_batches_per_epoch': len(train_loader),
        'val_batches_per_epoch': len(val_loader),
        'measurement': (('full epoch' if args.full_epoch else
                         f'extrapolated from {args.steps} train / {args.val_steps} val batches')
                        + f' · median of {max(args.repeats, 1)} pass(es)'
                        + ' · identical batch order across models'),
        'models': results,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'model':<12}{'params':>14}{'train/epoch':>14}{'val/epoch':>12}{'epoch':>12}{'vs fastest':>12}")
    fastest = min(r['epoch_s'] for r in results.values())
    for name, r in sorted(results.items(), key=lambda kv: kv[1]['epoch_s']):
        print(f"{name:<12}{r['params']:>14,}{r['train_epoch_s']/60:>13.1f}m"
              f"{r['val_epoch_s']/60:>11.1f}m{r['epoch_s']/60:>11.1f}m"
              f"{r['epoch_s']/fastest:>11.2f}x")
    print(f"\n💾 Wrote {args.out}")


if __name__ == '__main__':
    main()
