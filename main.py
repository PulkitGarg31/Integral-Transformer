#!/usr/bin/env python
"""Integral Legal Transformer — single entry point.

Run inside tmux on the GPU box:
    python main.py                     # data -> train -> eval-sc -> eval-delhi
    python main.py --stage train       # one stage only
    python main.py --stage smoke       # quick architecture sanity check
"""

import argparse
import datetime
import os
import sys


class Tee:
    """Mirror everything written to a stream into a log file too (like Unix `tee`).

    Used so every run leaves a permanent log in logs/ — the successor to the
    old hand-captured complete*_clean.txt files in legacy/.
    """

    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, text):
        self.stream.write(text)
        self.logfile.write(text)
        self.logfile.flush()   # flush every write so a crash/kill loses nothing
        return len(text)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        # Libraries probe sys.stdout for the full stream protocol (isatty,
        # fileno, encoding, ...) — delegate anything Tee doesn't wrap itself.
        return getattr(self.stream, name)


def setup_logging(stage, train_mode):
    """Mirror stdout into logs/run_<stage>_<mode>_<timestamp>.log; return the path."""
    os.makedirs("logs", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join("logs", f"run_{stage}_{train_mode}_{stamp}.log")
    logfile = open(path, "w", encoding="utf-8")
    # Legacy Windows consoles (cp1252) crash on the emoji in progress
    # messages; degrade unencodable characters instead of dying at startup.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.stdout = Tee(sys.stdout, logfile)
    print(f"📝 Console output is being saved to: {path}")
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', default='all',
                   choices=['all', 'data', 'train', 'eval-sc', 'eval-delhi', 'smoke'])
    p.add_argument('--train-mode', default='hidden', choices=['hidden', 'full'],
                   help='hidden = verdict-hidden training (paper regime), full = whole documents')
    p.add_argument('--n-remove', type=int, default=1,
                   help='chunks removed from document END in hidden-mode training')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--sc-path', default=None, help='Supreme Court PDF root')
    p.add_argument('--delhi-path', default=None, help='Delhi High Court PDF root')
    p.add_argument('--seed', type=int, default=16911)
    p.add_argument('--random-seed', action='store_true')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    return p.parse_args()


def build_config(args):
    from src.config import Config, set_seed
    config = Config()
    if args.sc_path:
        config.dataset_path = args.sc_path
    if args.delhi_path:
        config.delhi_path = args.delhi_path
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.resume_checkpoint = args.resume
    set_seed(None if args.random_seed else args.seed)
    return config


def require_path(path, what):
    if not os.path.isdir(path):
        sys.exit(f"ERROR: {what} not found at '{path}'. "
                 f"Pass the correct path with --sc-path / --delhi-path.")


# Built/loaded once per process — with --stage all, every stage reuses these
# instead of re-reading the multi-GB torch caches from disk.
_SC_DATASETS = None
_DELHI_DATASET = None


def load_or_build_sc_datasets(config, args):
    """Return (train_ds, val_ds, test_ds) for Supreme Court, using the v3 cache."""
    global _SC_DATASETS
    if _SC_DATASETS is not None:
        return _SC_DATASETS
    import torch
    from src.preprocessing import DataController, load_supreme_court_dataset
    from src.dataset import PreprocessedLegalDataset, dataset_fingerprint, sc_cache_name

    require_path(config.dataset_path, "Supreme Court dataset")
    controller = DataController(max_files=None, train_frac=config.train_split,
                                val_frac=config.val_split,
                                test_frac=config.test_split, seed=args.seed)
    pdf_paths = controller.discover_pdfs(config.dataset_path)   # walked ONCE
    cache = sc_cache_name(len(pdf_paths), dataset_fingerprint(config, controller))
    if os.path.exists(cache):
        print(f"🔥 Loading SC dataset cache: {cache}")
        d = torch.load(cache, weights_only=False)
        _SC_DATASETS = (d['train'], d['val'], d['test'])
        return _SC_DATASETS
    from src.models import load_tokenizer
    tokenizer = load_tokenizer(config.pretrained_model)
    tr_t, tr_l, va_t, va_l, te_t, te_l = load_supreme_court_dataset(
        config, controller, pdf_paths=pdf_paths)
    train = PreprocessedLegalDataset(tr_t, tr_l, tokenizer, config)
    val = PreprocessedLegalDataset(va_t, va_l, tokenizer, config)
    test = PreprocessedLegalDataset(te_t, te_l, tokenizer, config)
    torch.save({'train': train, 'val': val, 'test': test}, cache)
    print(f"💾 Saved SC cache: {cache}")
    _SC_DATASETS = (train, val, test)
    return _SC_DATASETS


def load_or_build_delhi_dataset(config, args):
    """Return the full Delhi HC dataset (all docs as test), using the v3 cache."""
    global _DELHI_DATASET
    if _DELHI_DATASET is not None:
        return _DELHI_DATASET
    import torch
    from src.preprocessing import DataController, load_supreme_court_dataset
    from src.dataset import PreprocessedLegalDataset, dataset_fingerprint, delhi_cache_name

    require_path(config.delhi_path, "Delhi High Court dataset")
    controller = DataController(max_files=None, train_frac=0.0, val_frac=0.0,
                                test_frac=1.0, seed=args.seed)
    pdf_paths = controller.discover_pdfs(config.delhi_path)     # walked ONCE
    cache = delhi_cache_name(len(pdf_paths), dataset_fingerprint(config, controller))
    if os.path.exists(cache):
        print(f"🔥 Loading Delhi dataset cache: {cache}")
        ds = torch.load(cache, weights_only=False)
    else:
        from src.models import load_tokenizer
        tokenizer = load_tokenizer(config.pretrained_model)
        # dataset_root is passed explicitly — shared config is never mutated
        _, _, _, _, te_t, te_l = load_supreme_court_dataset(
            config, controller, dataset_root=config.delhi_path, pdf_paths=pdf_paths)
        ds = PreprocessedLegalDataset(te_t, te_l, tokenizer, config)
        torch.save(ds, cache)
        print(f"💾 Saved Delhi cache: {cache}")
    _DELHI_DATASET = ds
    return ds


def stage_data(config, args):
    load_or_build_sc_datasets(config, args)
    load_or_build_delhi_dataset(config, args)


def stage_train(config, args):
    import gc
    import torch
    from src.dataset import ChunkRemovedDataset, make_dataloader
    from src.models import MODEL_NAMES, MODEL_REGISTRY
    from src.training import train_model, ckpt_prefix
    from src.evaluation import plot_training_curves

    train_ds, val_ds, _ = load_or_build_sc_datasets(config, args)
    if args.train_mode == 'hidden':
        print(f"🔪 Verdict-hidden training: removing last {args.n_remove} chunk(s)")
        train_ds = ChunkRemovedDataset(train_ds, args.n_remove)
        val_ds = ChunkRemovedDataset(val_ds, args.n_remove)
    train_loader = make_dataloader(train_ds, config.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, config.batch_size, shuffle=False)
    for name in MODEL_NAMES:
        model = MODEL_REGISTRY[name](config)
        train_model(model, name, config, train_loader, val_loader,
                    args.train_mode, args.n_remove)
        del model; torch.cuda.empty_cache(); gc.collect()
    plot_training_curves(ckpt_prefix(args.train_mode, args.n_remove))


def stage_eval_sc(config, args):
    from src.dataset import ChunkRemovedDataset, make_dataloader
    from src.evaluation import evaluate_checkpoints, print_comparison, save_plots
    from src.models import MODEL_NAMES

    _, _, test_ds = load_or_build_sc_datasets(config, args)
    if args.train_mode == 'hidden':
        test_ds = ChunkRemovedDataset(test_ds, args.n_remove)
    if len(test_ds) == 0:
        print(f"⚠️  SC TEST (n_remove={args.n_remove}): no documents left "
              f"after chunk removal — skipping.")
        return
    loader = make_dataloader(test_ds, config.batch_size, shuffle=False)
    results = evaluate_checkpoints(
        MODEL_NAMES, config, loader, args.train_mode, args.n_remove,
        title=f"SC TEST RESULTS (mode={args.train_mode}, n_remove={args.n_remove})")
    print_comparison(results, MODEL_NAMES)
    save_plots(results, f"sc_{args.train_mode}{args.n_remove if args.train_mode == 'hidden' else ''}")


def stage_eval_delhi(config, args):
    from src.dataset import ChunkRemovedDataset, make_dataloader
    from src.evaluation import evaluate_checkpoints, print_comparison, save_plots
    from src.models import MODEL_NAMES

    delhi = load_or_build_delhi_dataset(config, args)
    runs = [("DELHI — FULL DOCUMENTS", "delhi_full", delhi)]
    for k in (1, 2):
        runs.append((f"DELHI — LAST {k} CHUNK(S) REMOVED", f"delhi_hidden{k}",
                     ChunkRemovedDataset(delhi, k)))
    for title, tag, ds in runs:
        if len(ds) == 0:
            print(f"⚠️  {title}: no documents left — skipping.")
            continue
        loader = make_dataloader(ds, 8, shuffle=False)   # small eval batch (legacy test.py)
        results = evaluate_checkpoints(MODEL_NAMES, config, loader,
                                       args.train_mode, args.n_remove, title=title)
        print_comparison(results, MODEL_NAMES)
        save_plots(results, tag)


def stage_smoke(config, args):
    """Tiny forward pass of every architecture (adapted from legacy quick_test)."""
    import torch
    from src.models import MODEL_NAMES, MODEL_REGISTRY
    config.max_chunks = 4
    for name in MODEL_NAMES:
        model = MODEL_REGISTRY[name](config)
        ids = torch.randint(0, 1000, (2, 4, 512))
        mask = torch.ones(2, 4, 512, dtype=torch.long)
        cmask = torch.ones(2, 4)
        with torch.no_grad():
            logits = model(ids, mask, cmask)
        assert logits.shape == (2, config.num_classes), f"{name}: bad shape {logits.shape}"
        print(f"✓ {name}: forward pass OK {tuple(logits.shape)}")
        del model
    print("\n✓ Smoke test passed for all 4 architectures.")


def main():
    args = parse_args()          # BEFORE src imports so --help works anywhere
    setup_logging(args.stage, args.train_mode)   # every run leaves a log file
    config = build_config(args)
    stages = {'data': stage_data, 'train': stage_train, 'eval-sc': stage_eval_sc,
              'eval-delhi': stage_eval_delhi, 'smoke': stage_smoke}
    if args.stage == 'all':
        for s in ('data', 'train', 'eval-sc', 'eval-delhi'):
            print(f"\n{'#' * 70}\n# STAGE: {s}\n{'#' * 70}")
            stages[s](config, args)
    else:
        stages[args.stage](config, args)


if __name__ == "__main__":
    main()
