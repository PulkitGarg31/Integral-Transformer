"""Single training loop shared by all models and both regimes (full / verdict-hidden)."""

import gc
import json
import os

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.evaluation import evaluate_model, history_path


def ckpt_prefix(mode, n_remove):
    return "full" if mode == "full" else f"hidden{n_remove}"


def ckpt_names(mode, n_remove, model_name, save_dir=None):
    p = ckpt_prefix(mode, n_remove)
    best, latest = f"{p}_{model_name}_best.pt", f"{p}_{model_name}_latest.pt"
    if save_dir:
        best = os.path.join(save_dir, best)
        latest = os.path.join(save_dir, latest)
    return best, latest


def build_optimizer(model, model_name, config):
    if model_name == 'integral':
        # Group by module REFERENCE, not name substring: renaming a module or
        # adding one whose name merely contains 'integration' must not
        # silently change which learning rate its parameters get.
        enc = list(model.encoder.parameters())
        integ = list(model.integration_layers.parameters())
        grouped = {id(p) for p in enc} | {id(p) for p in integ}
        other = [p for p in model.parameters() if id(p) not in grouped]
        return torch.optim.AdamW([
            {'params': enc, 'lr': config.learning_rate},
            {'params': integ, 'lr': config.integration_lr},
            {'params': other, 'lr': config.learning_rate},
        ], weight_decay=config.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                             weight_decay=config.weight_decay)


def train_model(model, model_name, config, train_loader, val_loader, mode, n_remove):
    """Train one model with early stopping; skip if best checkpoint exists.

    Copied from legacy/predict_train.py train_single_model (lines 103-208)
    with only these edits:
      - checkpoint names come from ckpt_names(mode, n_remove, model_name)
      - optimizer comes from build_optimizer()
      - validation uses evaluate_model() from src.evaluation
      - device / use_fp16 come from config.device / config.use_fp16
    The epoch loop, AMP/grad-accum logic, save/resume dict structure, and
    early-stopping logic are otherwise unchanged.
    """
    device = config.device
    use_fp16 = getattr(config, 'use_fp16', False) and device == 'cuda'
    save_dir = getattr(config, 'save_dir', None)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    ckpt_best, ckpt_latest = ckpt_names(mode, n_remove, model_name, save_dir)

    # Skip if already trained — UNLESS a resume was requested: the best
    # checkpoint appears at the first improving epoch, so checking it first
    # made --resume dead code and silently truncated interrupted runs.
    resuming = config.resume_checkpoint and os.path.exists(ckpt_latest)
    if os.path.exists(ckpt_best) and not resuming:
        print(f"\n⏭️  SKIPPING {model_name.upper()} — '{ckpt_best}' exists. Delete to retrain.")
        return model

    model.to(device)

    optimizer = build_optimizer(model, model_name, config)

    accum = config.gradient_accumulation_steps
    total_steps = len(train_loader) * config.num_epochs // accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(config, 'label_smoothing', 0.0))
    scaler = GradScaler('cuda', enabled=use_fp16)

    best_f1 = 0
    patience = getattr(config, 'early_stopping_patience', 999)
    no_improve = 0
    start_epoch = 0

    # Per-epoch history — recorded so training curves can be plotted later
    # (plot_training_curves) without retraining. Written to disk every epoch.
    hist_file = history_path(ckpt_prefix(mode, n_remove), model_name)
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}

    # Resume logic
    if resuming:
        print(f"  \U0001f504 Resuming from {ckpt_latest}...")
        try:
            # A crash mid torch.save leaves a truncated latest checkpoint —
            # exactly the situation --resume is for, so fall back to fresh
            # training instead of crashing on it.
            ck = torch.load(ckpt_latest, map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scheduler.load_state_dict(ck['scheduler_state_dict'])
            scaler.load_state_dict(ck['scaler_state_dict'])
            start_epoch = ck['epoch'] + 1
            best_f1 = ck['best_val_f1']
            no_improve = ck.get('no_improve', 0)   # early-stopping patience persists
            del ck; gc.collect()
        except Exception as e:
            print(f"  ⚠️  Could not resume from '{ckpt_latest}' ({e}) — starting fresh.")
            start_epoch, best_f1, no_improve = 0, 0, 0
            resuming = False
            gc.collect()
        if resuming and no_improve >= patience:
            print(f"  ⏹ Early stopping already triggered ({no_improve}/{patience}) — nothing to resume.")
            return model
        if start_epoch >= config.num_epochs:
            print(f"  ✓ Training already completed ({start_epoch}/{config.num_epochs} epochs).")
            return model
        if os.path.exists(hist_file):
            with open(hist_file, encoding='utf-8') as f:
                prev = json.load(f)
            # keep only epochs before the resume point, in case of a crash mid-save
            history = {k: prev.get(k, [])[:start_epoch] for k in history}

    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()} (FP16={use_fp16}, params={sum(p.numel() for p in model.parameters()):,})")
    print(f"{'='*60}")

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False)):
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            cmask = batch['chunk_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            with autocast('cuda', enabled=use_fp16):
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
            total_loss += loss.item() * accum

        train_loss = total_loss / max(len(train_loader), 1)
        vl, vf1, vacc, _, _ = evaluate_model(model, val_loader, device, use_fp16, criterion)

        print(f"  Epoch {epoch+1}/{config.num_epochs} — "
              f"Train Loss={train_loss:.4f}, Val Loss={vl:.4f}, Val F1={vf1:.4f}, Val Acc={vacc:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(vl)
        history['val_f1'].append(vf1)
        history['val_acc'].append(vacc)
        os.makedirs(os.path.dirname(hist_file), exist_ok=True)
        with open(hist_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        # Update best/patience BEFORE building the checkpoint state, so a
        # resumed run restores the true best F1 and the patience counter —
        # a stale best_val_f1 let later, worse epochs overwrite the real best.
        improved = vf1 > best_f1 + 1e-4
        if improved:
            best_f1 = vf1
            no_improve = 0
        else:
            no_improve += 1

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_f1': best_f1,
            'no_improve': no_improve,
        }
        torch.save(state, ckpt_latest)

        if improved:
            torch.save(state, ckpt_best)
            print(f"  ✓ New best! F1={vf1:.4f}")
        else:
            print(f"  ↔ No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

    return model
