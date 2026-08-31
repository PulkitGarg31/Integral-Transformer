#!/usr/bin/env python
"""Ablation study for the Integral model (paper Table V).

Trains each ablated variant FROM SCRATCH under the exact conditions of the
main verdict-hidden run (same data, seed, epochs, patience, optimizer LR
groups), then evaluates every variant on the SC hidden test set right after
its training and prints Table-V-style comparisons. The Delhi HC transfer
column lives in ablation_delhi.py, which can run concurrently on the same
GPU and picks up each variant as soon as its .done marker appears.

Variants:
  full               — reference. Reuses checkpoints/hidden1_integral_best.pt
                       from main.py if present (trains it if not).
  no_adaptive_gate   — AdaptiveGate replaced by a fixed 50/50 attn/integ mix.
  single_scale       — multi-scale RBF kernel reduced to the fine scale only
                       (scale_mix pinned so sigmoid(mix) ~= 1, frozen).
  no_pos_bias        — kernel position bias frozen at zero (never learned).
  no_label_smoothing — trained with label_smoothing = 0.
  no_dropout         — integration-kernel dropout off (integration_dropout=0),
                       i.e. the v2 kernel; isolates the run-2 regularisation.

Run from `objective 1/` on the GPU box AFTER the main run finishes
(they share the GPU and the full-integral checkpoint):

    python ablation.py                  # train all variants, SC test each
    python ablation.py --eval both      # ...and Delhi full inline (old behaviour)
    python ablation.py --variants no_dropout,no_adaptive_gate   # a subset
    python ablation.py --skip-train     # evaluate existing checkpoints only
    python ablation.py --smoke          # CPU sanity check of variant surgery

    # in a second shell, while the above trains:
    python ablation_delhi.py --watch    # Delhi full for every finished variant

Variants train in ABLATIONS order; 'full' is always evaluated (it costs no
training) so Δ F1 has a reference. Each variant checkpoints as
checkpoints/hidden1_abl_<variant>_best.pt and writes
checkpoints/hidden1_abl_<variant>.done once its training has finished (the
_best.pt file is rewritten at every improving epoch, so the marker is what
tells ablation_delhi.py a checkpoint is final).

Stopping and rerunning is safe: finished variants (marker present) skip
training and are just re-tested; a variant that was mid-training when the
run was killed (latest checkpoint present, no marker) is RESUMED from its
last epoch automatically — a plain rerun never mistakes its partial
_best.pt for a final one. Delete a variant's *_best.pt/*_latest.pt to
retrain it from scratch. Results are merged into ablation_results.json
after every variant, so nothing is lost on interruption.
"""
import argparse
import copy
import datetime
import gc
import json
import os
import types

import torch
import torch.nn as nn

from main import (load_or_build_delhi_dataset, load_or_build_sc_datasets,
                  setup_logging)
from src.config import Config, set_seed
from src.dataset import ChunkRemovedDataset, make_dataloader
from src.models import IntegralTransformerModel
import src.training as training

MODE, N_REMOVE = "hidden", 1     # same regime as the main paper run

ABLATIONS = {
    "full": "Full Integral",
    "no_adaptive_gate": "No adaptive gate",
    "single_scale": "Single-scale kernel",
    "no_pos_bias": "No position bias",
    "no_label_smoothing": "No label smoothing",
    "no_dropout": "No kernel dropout",
}
CONFIG_ONLY = ("full", "no_dropout", "no_label_smoothing")   # no model surgery


class FixedGate(nn.Module):
    """Replaces AdaptiveGate: constant 0.5 -> static mean of attn and integ."""

    def forward(self, attn, integ):
        return attn.new_tensor(0.5)


def make_variant(variant, config):
    """Build an IntegralTransformerModel and apply the ablation surgery."""
    model = IntegralTransformerModel(config)
    if variant == "single_scale":
        for kernel in model.integration_layers:
            with torch.no_grad():
                kernel.scale_mix.fill_(20.0)      # sigmoid(20) ~= 1: fine only
            kernel.scale_mix.requires_grad_(False)
    elif variant == "no_pos_bias":
        for kernel in model.integration_layers:
            with torch.no_grad():
                kernel.pos_bias.weight.zero_()
            kernel.pos_bias.weight.requires_grad_(False)
    elif variant == "no_adaptive_gate":
        model.gates = nn.ModuleList(FixedGate() for _ in model.gates)
    elif variant not in CONFIG_ONLY:
        raise ValueError(f"unknown variant '{variant}'")
    return model


def variant_config(variant, config):
    cfg = copy.copy(config)
    if variant == "no_label_smoothing":
        cfg.label_smoothing = 0.0
    elif variant == "no_dropout":
        cfg.integration_dropout = 0.0        # nn.Dropout(0.0) == identity
    return cfg


def ckpt_model_name(variant):
    # 'full' IS the main model — share its checkpoint instead of retraining.
    return "integral" if variant == "full" else f"abl_{variant}"


def done_marker(variant, save_dir=None):
    """Path of the marker written once a variant's training has finished.

    ablation_delhi.py evaluates a variant only after this exists — the
    _best.pt checkpoint is rewritten at every improving epoch, so its mere
    presence does not mean training is over.
    """
    prefix = training.ckpt_prefix(MODE, N_REMOVE)
    name = f"{prefix}_{ckpt_model_name(variant)}.done"
    return os.path.join(save_dir, name) if save_dir else name


def write_done_marker(variant, best_path, save_dir=None):
    path = done_marker(variant, save_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "checkpoint": best_path,
                   "finished_at": datetime.datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)
    return path


# Every variant keeps integration_layers at the higher integration_lr, exactly
# like the main integral model — otherwise the optimizer would become a
# confounder. build_optimizer keys on the name, so route variants through the
# 'integral' branch (FixedGate has no params; 'other' group absorbs nothing new).
_orig_build_optimizer = training.build_optimizer


def _ablation_build_optimizer(model, model_name, config):
    if hasattr(model, "integration_layers"):
        model_name = "integral"
    return _orig_build_optimizer(model, model_name, config)


training.build_optimizer = _ablation_build_optimizer


RESULTS_FILE = "ablation_results.json"


def load_results():
    """Previous rows (a rerun or a --variants subset must not wipe them)."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def eval_variant(model, loader, device, use_fp16):
    from sklearn.metrics import classification_report
    from src.evaluation import evaluate_model
    criterion = nn.CrossEntropyLoss()          # plain CE for comparable losses
    loss, f1, acc, preds, labels = evaluate_model(
        model, loader, device, use_fp16, criterion)
    print(classification_report(labels, preds,
                                target_names=["Rejected", "Accepted"],
                                digits=4, zero_division=0))
    return {"loss": loss, "f1": f1, "acc": acc}


def print_table(title, results):
    ref = results.get("full", {}).get(title)
    print(f"\n{'=' * 70}\nABLATION — {title.upper()}\n{'=' * 70}")
    print(f"{'Ablation':<22} {'Loss':>8} {'Macro-F1':>10} {'Acc':>8} {'Δ F1':>9}")
    print("-" * 62)
    for variant, label in ABLATIONS.items():
        r = results.get(variant, {}).get(title)
        if not r:
            print(f"{label:<22} {'—':>8}")
            continue
        delta = r["f1"] - ref["f1"] if ref else 0.0
        print(f"{label:<22} {r['loss']:>8.4f} {r['f1']:>10.4f} "
              f"{r['acc']:>8.4f} {delta:>+9.4f}")


def run_study(args):
    config = Config()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.resume_checkpoint = args.resume
    device = config.device
    use_fp16 = getattr(config, "use_fp16", False) and device == "cuda"

    ns = types.SimpleNamespace(seed=args.seed)
    train_ds, val_ds, test_ds = load_or_build_sc_datasets(config, ns)
    train_ds = ChunkRemovedDataset(train_ds, N_REMOVE)
    val_ds = ChunkRemovedDataset(val_ds, N_REMOVE)
    test_ds = ChunkRemovedDataset(test_ds, N_REMOVE)

    train_loader = make_dataloader(train_ds, config.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, config.batch_size, shuffle=False)
    sc_loader = make_dataloader(test_ds, config.batch_size, shuffle=False)
    delhi_loader = None
    if args.eval == "both":
        delhi_ds = load_or_build_delhi_dataset(config, ns)
        delhi_loader = make_dataloader(delhi_ds, 8, shuffle=False)  # legacy eval batch
    else:
        print("ℹ️  Delhi HC eval not run here — use `python ablation_delhi.py --watch` "
              "in a second shell (it can share the GPU).")

    results = load_results()
    selected = [v for v in ABLATIONS if v in args.variants or v == "full"]
    for variant in selected:
        cfg = variant_config(variant, config)
        name = ckpt_model_name(variant)
        save_dir = getattr(config, "save_dir", None)
        best, latest = training.ckpt_names(MODE, N_REMOVE, name, save_dir)
        if not args.skip_train:
            # A killed run leaves _latest.pt (and usually _best.pt) but no
            # .done marker. Without resume, train_model's skip-if-exists would
            # accept that partial _best.pt as final — so resume it instead.
            # 'full' is main.py's checkpoint and complete by contract.
            interrupted = (variant != "full" and os.path.exists(latest)
                           and not os.path.exists(done_marker(variant, save_dir)))
            cfg.resume_checkpoint = args.resume or interrupted
            if interrupted and not args.resume:
                print(f"🔁 {variant}: found an interrupted training run "
                      f"({latest}, no .done marker) — resuming it.")
            # Same seed before EVERY variant: identical init draws and batch
            # order, so the ablation is the only difference between runs.
            set_seed(args.seed)
            model = make_variant(variant, cfg)
            training.train_model(model, name, cfg, train_loader, val_loader,
                                 MODE, N_REMOVE)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            if os.path.exists(best):
                # train_model has returned (early-stopped, ran out of epochs,
                # or skipped because best already existed): best is final.
                marker = write_done_marker(variant, best, getattr(config, "save_dir", None))
                print(f"✅ {variant}: training finished — marker {marker}")

        if not os.path.exists(best):
            print(f"⚠️  {variant}: '{best}' not found — skipping eval.")
            continue
        print(f"\n{'#' * 70}\n# EVAL: {ABLATIONS[variant]}  ({best})\n{'#' * 70}")
        model = make_variant(variant, cfg)
        ck = torch.load(best, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
        del ck
        model.to(device)
        results.setdefault(variant, {})
        print(f"\n--- SC test, verdict hidden (n_remove={N_REMOVE}) ---")
        results[variant]["sc_hidden"] = eval_variant(model, sc_loader, device, use_fp16)
        if delhi_loader is not None:
            print("\n--- Delhi HC, full documents (cross-dataset) ---")
            results[variant]["delhi_full"] = eval_variant(model, delhi_loader, device, use_fp16)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        save_results(results)                  # persist after EVERY variant

    if delhi_loader is not None:
        print_table("delhi_full", results)     # the paper's Table V
    print_table("sc_hidden", results)
    save_results(results)
    print(f"\n💾 Results written to {RESULTS_FILE}")


def smoke():
    """CPU check of the variant surgery — run locally before spending GPU time."""
    from transformers import BertConfig, BertModel
    import src.models as models
    models.load_encoder = lambda name: BertModel(BertConfig(
        vocab_size=200, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128,
        max_position_embeddings=64))

    cfg = Config()
    cfg.hidden_size = 64
    cfg.max_chunks = 4
    cfg.integration_kernel_size = 16

    ids = torch.randint(0, 200, (2, 4, 16))
    mask = torch.ones(2, 4, 16, dtype=torch.long)
    cmask = torch.ones(2, 4)
    for variant in ABLATIONS:
        model = make_variant(variant, variant_config(variant, cfg))
        want_p = 0.0 if variant == "no_dropout" else cfg.integration_dropout
        for k in model.integration_layers:
            assert k.weight_dropout.p == want_p and k.output_dropout.p == want_p, (
                f"{variant}: kernel dropout p={k.weight_dropout.p}, want {want_p}")
        if variant == "single_scale":
            for k in model.integration_layers:
                assert torch.sigmoid(k.scale_mix).min() > 0.999
                assert not k.scale_mix.requires_grad
        if variant == "no_pos_bias":
            for k in model.integration_layers:
                assert k.pos_bias.weight.abs().sum() == 0
                assert not k.pos_bias.weight.requires_grad
        if variant == "no_adaptive_gate":
            assert all(isinstance(g, FixedGate) for g in model.gates)
        opt = training.build_optimizer(model, ckpt_model_name(variant), cfg)
        assert len(opt.param_groups) == 3, f"{variant}: optimizer groups wrong"
        with torch.no_grad():
            logits = model(ids, mask, cmask)
        assert logits.shape == (2, cfg.num_classes)
        assert variant_config(variant, cfg).label_smoothing == \
            (0.0 if variant == "no_label_smoothing" else cfg.label_smoothing)
        print(f"✓ {variant}: surgery + forward pass + optimizer OK")
    print(f"\n✓ Smoke test passed for all {len(ABLATIONS)} variants.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=16911)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-train", action="store_true",
                   help="evaluate existing ablation checkpoints only")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--variants", default=",".join(ABLATIONS),
                   help="comma-separated subset to train/eval (default: all); "
                        "'full' is always included as the reference")
    p.add_argument("--eval", choices=["sc", "both"], default="sc",
                   help="sc: SC hidden test after each variant (default; Delhi is "
                        "done by ablation_delhi.py); both: also Delhi full inline")
    args = p.parse_args()
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in args.variants if v not in ABLATIONS]
    if unknown:
        p.error(f"unknown variant(s) {unknown}; choose from {list(ABLATIONS)}")
    if args.smoke:
        smoke()
        return
    setup_logging("ablation", MODE)
    set_seed(args.seed)
    run_study(args)


if __name__ == "__main__":
    main()
