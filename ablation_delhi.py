#!/usr/bin/env python
"""Delhi HC full-document evaluation of the ablation checkpoints (paper Table V,
cross-dataset column).

Runs as its own process so it can share the GPU with ablation.py while the
next variant trains. Inference only, default batch 4 (~1-2 GB); on a CUDA
OOM it halves the batch and retries down to 1 instead of dying, so it can
never take itself down and it never asks for a big allocation the trainer
might need. Metrics are batch-size independent (per-document F1/acc; loss
is a mean of equal-size batch means — 53,352 docs divide evenly by 8/4/2/1).

    python ablation_delhi.py --watch          # poll; evaluate variants as they
                                              # finish; exit when all are done
    python ablation_delhi.py                  # one pass over what is finished
    python ablation_delhi.py --variants no_dropout,no_adaptive_gate
    python ablation_delhi.py --assume-done    # any existing *_best.pt, no marker
    python ablation_delhi.py --force          # re-evaluate already-done variants

A variant is evaluated only once its training is FINISHED:
  * ablated variants: ablation.py writes checkpoints/hidden1_abl_<v>.done after
    train_model returns. The _best.pt file alone is not enough — it is
    rewritten at every improving epoch while training is still running.
  * full: checkpoints/hidden1_integral_best.pt from main.py, which must have
    completed before the ablation study started (it has, by construction:
    ablation.py needs it too).

Results accumulate in ablation_delhi_results.json (rerun-safe). If
ablation_results.json from ablation.py exists, a combined SC + Delhi Table-V
view is printed at the end.
"""
import argparse
import gc
import json
import os
import time
import types

import torch

from ablation import (ABLATIONS, MODE, N_REMOVE, ckpt_model_name, done_marker,
                      eval_variant, make_variant, print_table, variant_config)
from main import load_or_build_delhi_dataset, setup_logging
from src.config import Config, set_seed
from src.dataset import make_dataloader
import src.training as training

RESULTS_FILE = "ablation_delhi_results.json"
SC_RESULTS_FILE = "ablation_results.json"       # written by ablation.py


def best_ckpt(variant, config):
    best, _ = training.ckpt_names(MODE, N_REMOVE, ckpt_model_name(variant),
                                  getattr(config, "save_dir", None))
    return best


def is_finished(variant, config, assume_done=False):
    """True when the variant's best checkpoint exists AND is final."""
    if not os.path.exists(best_ckpt(variant, config)):
        return False
    if assume_done or variant == "full":
        return True
    return os.path.exists(done_marker(variant, getattr(config, "save_dir", None)))


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def evaluate(variant, config, delhi_ds, batch_size, device, use_fp16):
    best = best_ckpt(variant, config)
    print(f"\n{'#' * 70}\n# DELHI EVAL: {ABLATIONS[variant]}  ({best})\n{'#' * 70}")
    model = make_variant(variant, variant_config(variant, config))
    ck = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    del ck
    model.to(device)
    try:
        bs = batch_size
        while True:
            try:
                print(f"\n--- Delhi HC, full documents (cross-dataset), batch={bs} ---")
                loader = make_dataloader(delhi_ds, bs, shuffle=False)
                return eval_variant(model, loader, device, use_fp16)
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                torch.cuda.empty_cache()
                gc.collect()
                print(f"⚠️  CUDA OOM at batch {bs} (GPU shared with training?) — retrying at {bs // 2}")
                bs //= 2
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def print_combined(delhi_results):
    """Side-by-side SC (from ablation.py) + Delhi (from here) — the Table V view."""
    if not os.path.exists(SC_RESULTS_FILE):
        return
    with open(SC_RESULTS_FILE, encoding="utf-8") as f:
        sc = json.load(f)
    ref_sc = sc.get("full", {}).get("sc_hidden")
    ref_de = delhi_results.get("full", {}).get("delhi_full")
    print(f"\n{'=' * 78}\nABLATION — SC HIDDEN vs DELHI FULL (Table V view)\n{'=' * 78}")
    print(f"{'Ablation':<22} {'SC F1':>8} {'Δ SC':>8}   {'Delhi F1':>9} {'Δ Delhi':>9}")
    print("-" * 62)
    for v, label in ABLATIONS.items():
        s = sc.get(v, {}).get("sc_hidden")
        d = delhi_results.get(v, {}).get("delhi_full")
        s_txt = f"{s['f1']:>8.4f} {s['f1'] - ref_sc['f1']:>+8.4f}" if s and ref_sc else f"{'—':>8} {'':>8}"
        d_txt = f"{d['f1']:>9.4f} {d['f1'] - ref_de['f1']:>+9.4f}" if d and ref_de else f"{'—':>9} {'':>9}"
        print(f"{label:<22} {s_txt}   {d_txt}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=16911,
                   help="must match the ablation/main run (selects the dataset cache)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Delhi eval batch (default 4: small on purpose, the GPU is shared "
                        "with training; halves automatically on OOM)")
    p.add_argument("--variants", default=",".join(ABLATIONS),
                   help="comma-separated subset (default: all); 'full' always included")
    p.add_argument("--watch", action="store_true",
                   help="keep polling until every selected variant is evaluated")
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--assume-done", action="store_true",
                   help="evaluate any existing *_best.pt without waiting for its .done marker "
                        "(only when you KNOW no variant is still training)")
    p.add_argument("--force", action="store_true",
                   help="re-evaluate variants already present in the results file")
    args = p.parse_args()
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in args.variants if v not in ABLATIONS]
    if unknown:
        p.error(f"unknown variant(s) {unknown}; choose from {list(ABLATIONS)}")
    selected = [v for v in ABLATIONS if v in args.variants or v == "full"]

    setup_logging("ablation_delhi", MODE)
    set_seed(args.seed)
    config = Config()
    device = config.device
    use_fp16 = getattr(config, "use_fp16", False) and device == "cuda"

    ns = types.SimpleNamespace(seed=args.seed)
    delhi_ds = load_or_build_delhi_dataset(config, ns)

    results = load_results()
    pending = [v for v in selected if args.force or "delhi_full" not in results.get(v, {})]
    already = [v for v in selected if v not in pending]
    if already:
        print(f"↩️  already evaluated (use --force to redo): {already}")

    while pending:
        progressed = False
        for v in list(pending):
            if not is_finished(v, config, args.assume_done):
                continue
            results.setdefault(v, {})["delhi_full"] = evaluate(
                v, config, delhi_ds, args.batch_size, device, use_fp16)
            save_results(results)                       # persist after EVERY variant
            pending.remove(v)
            progressed = True
        if not pending:
            break
        if not args.watch:
            print(f"\n⏳ not finished yet (no .done marker / checkpoint): {pending}"
                  f"\n   rerun later, or use --watch to keep polling.")
            break
        if not progressed:
            print(f"⏳ waiting for {pending} — next check in {args.poll_seconds}s "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
        time.sleep(args.poll_seconds)

    print_table("delhi_full", results)                  # the paper's Table V column
    print_combined(results)
    print(f"\n💾 Results in {RESULTS_FILE}")


if __name__ == "__main__":
    main()
