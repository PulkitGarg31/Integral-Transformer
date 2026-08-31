"""Validation/evaluation helpers shared by all training and testing scripts.

Module-level imports are torch-free so print_comparison is importable/testable
without torch installed. Heavy imports (torch, sklearn) are deferred inside
function bodies.
"""


def print_comparison(final_results, model_names, ref='integral'):
    """Format and print the model comparison table + relative improvements.
    Logic copied from legacy/predict_test.py lines 240-266, returning the text."""
    lines = []
    if not final_results:
        return ""
    ref_name = ref if ref in final_results else next(iter(final_results))
    ir = final_results[ref_name]
    lines.append(f"\n{'Model':<20} {'Loss':>10} {'F1':>10} {'Acc':>10} "
                 f"{'Δ F1 vs ' + ref_name.upper():>20}")
    lines.append("-" * 74)
    for name in model_names:
        if name not in final_results:
            continue
        r = final_results[name]
        delta = r['f1'] - ir['f1'] if name != ref_name else 0.0
        marker = " ★" if name == ref_name else ""
        lines.append(f"  {name.upper():<18} {r['loss']:>10.4f} {r['f1']:>10.4f} "
                     f"{r['acc']:>10.4f} {delta:>+20.4f}{marker}")
    if ref_name == 'integral':
        lines.append("\n📈 Relative Improvement of INTEGRAL over baselines:")
        for bn in [n for n in model_names if n != ref_name]:
            if bn in final_results:
                br = final_results[bn]
                f1i = ((ir['f1'] - br['f1']) / br['f1']) * 100 if br['f1'] > 0 else 0
                aci = ((ir['acc'] - br['acc']) / br['acc']) * 100 if br['acc'] > 0 else 0
                lines.append(f"   vs {bn.upper():<12}: F1 {f1i:+.2f}%,  Acc {aci:+.2f}%")
    text = "\n".join(lines)
    print(text)
    return text


def evaluate_checkpoints(model_names, config, loader, mode, n_remove, title):
    """Load each best checkpoint, evaluate on loader, print reports, return results dict.
    Consolidates the per-model eval loops of legacy/test.py and legacy/predict_test.py."""
    import gc
    import os
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report
    from src.models import MODEL_REGISTRY
    from src.training import ckpt_names

    criterion = nn.CrossEntropyLoss()
    device = config.device
    use_fp16 = getattr(config, 'use_fp16', False) and device == 'cuda'
    final_results = {}
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    for name in model_names:
        best_file, _ = ckpt_names(mode, n_remove, name,
                                  save_dir=getattr(config, 'save_dir', None))
        if not os.path.exists(best_file):
            print(f"\n⚠️  {name.upper()}: '{best_file}' not found — skipping.")
            continue
        print(f"\nEvaluating {name.upper()} — checkpoint: {os.path.abspath(best_file)}")
        model = MODEL_REGISTRY[name](config)
        try:
            # One corrupt checkpoint must not abort the whole eval stage
            # (legacy test.py behavior): report it and evaluate the rest.
            ckpt = torch.load(best_file, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            del ckpt; gc.collect()
        except Exception as e:
            print(f"⚠️  Error loading {best_file}: {e} — skipping {name.upper()}.")
            del model; torch.cuda.empty_cache(); gc.collect()
            continue
        model.to(device)
        tl, tf1, ta, preds, labels = evaluate_model(model, loader, device, use_fp16, criterion)
        # preds/labels kept so save_plots can draw confusion matrices afterwards
        final_results[name] = {'loss': tl, 'f1': tf1, 'acc': ta,
                               'preds': list(preds), 'labels': list(labels)}
        print(f"  {name.upper()} → Loss={tl:.4f}, F1={tf1:.4f}, Acc={ta:.4f}")
        # labels=[0, 1] keeps single-class subsets (e.g. a small Delhi split
        # where the model predicts one class) from crashing sklearn.
        print(classification_report(labels, preds, labels=[0, 1],
                                    target_names=['Rejected', 'Accepted'],
                                    zero_division=0))
        del model; torch.cuda.empty_cache(); gc.collect()
    return final_results


def save_plots(final_results, tag, class_names=('Rejected', 'Accepted'),
               model_order=None):
    """Save evaluation visuals to plots/: one confusion-matrix grid and one
    F1/Accuracy bar chart per evaluation, named by `tag` (e.g. 'delhi_hidden1').

    Needs 'preds'/'labels' inside each model's results entry (added by
    evaluate_checkpoints). Skips gracefully if matplotlib/sklearn are missing
    so evaluation never crashes over a plotting library. Returns saved paths.
    """
    if not final_results:
        return []
    if model_order is None:
        from src.constants import MODEL_NAMES as model_order   # torch-free source of truth
    try:
        import matplotlib
        matplotlib.use('Agg')          # headless GPU box: no display needed
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix
    except ImportError as e:
        print(f"⚠️  Plotting skipped (library missing: {e.name}). "
              f"Install matplotlib + scikit-learn to get plots.")
        return []
    import os

    os.makedirs('plots', exist_ok=True)
    saved = []
    models = [m for m in model_order if m in final_results]

    # --- 1. Confusion matrix grid (one subplot per model) -----------------
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2))
    if len(models) == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        r = final_results[name]
        cm = confusion_matrix(r['labels'], r['preds'], labels=[0, 1])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=11)
        ax.set_title(f"{name.upper()}\nF1={r['f1']:.4f}  Acc={r['acc']:.4f}", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
        ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    fig.suptitle(f"Confusion Matrices — {tag}", fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join('plots', f"confusion_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)

    # --- 2. F1 / Accuracy comparison bar chart ----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35
    f1s = [final_results[m]['f1'] for m in models]
    accs = [final_results[m]['acc'] for m in models]
    bars1 = ax.bar(x - width / 2, f1s, width, label='Macro F1', color='steelblue')
    bars2 = ax.bar(x + width / 2, accs, width, label='Accuracy', color='coral')
    for bars in (bars1, bars2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylim(0, 1); ax.set_ylabel('Score')
    ax.set_title(f"Model Comparison — {tag}", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    path = os.path.join('plots', f"comparison_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)

    print(f"📊 Plots saved: {', '.join(saved)}")
    return saved


def history_path(prefix, model_name):
    """Path of a model's per-epoch training history JSON (see train_model)."""
    import os
    return os.path.join('history', f"{prefix}_{model_name}_history.json")


def plot_training_curves(prefix, model_order=None):
    """Draw the per-epoch training curves for every model whose history JSON
    exists (recorded by train_model), like legacy auto.py's plot_comparison.

    Saves plots/training_curves_<prefix>.png and returns its path, or None if
    there is no history or matplotlib is unavailable.
    """
    import json
    import os

    if model_order is None:
        from src.constants import MODEL_NAMES as model_order   # torch-free source of truth
    histories = {}
    for name in model_order:
        p = history_path(prefix, name)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                histories[name] = json.load(f)
    if not histories:
        print(f"⚠️  No training history found for '{prefix}' — nothing to plot.")
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  Plotting skipped (matplotlib missing). "
              "Install matplotlib to get training curves.")
        return None

    STYLES = {
        'integral':   ('steelblue', '-',  'Integral (Ours)'),
        'baseline':   ('coral',     '--', 'Baseline (Attn-Only)'),
        'truncation': ('seagreen',  '-.', 'Truncation (First 512)'),
        'meanpool':   ('orchid',    ':',  'Mean Pool (No Interaction)'),
    }
    PANELS = [('train_loss', 'Training Loss', 'Loss'),
              ('val_loss', 'Validation Loss', 'Loss'),
              ('val_f1', 'Validation F1', 'F1 Score'),
              ('val_acc', 'Validation Accuracy', 'Accuracy')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves — {prefix}", fontsize=14, fontweight='bold')
    for ax, (key, title, ylabel) in zip(axes.flat, PANELS):
        for name, hist in histories.items():
            c, ls, lbl = STYLES.get(name, ('gray', '-', name))
            values = hist.get(key, [])
            ax.plot(range(1, len(values) + 1), values,
                    color=c, linestyle=ls, label=lbl, linewidth=2)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs('plots', exist_ok=True)
    path = os.path.join('plots', f"training_curves_{prefix}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Training curves saved: {path}")
    return path


def evaluate_model(model, loader, device, use_fp16, criterion):
    """Run one full pass over `loader` in eval mode and return aggregate metrics.

    Copied verbatim from legacy/predict_train.py:76-96 (Task 7 extends this
    module with print_comparison and other reporting helpers). Heavy
    dependencies (torch, autocast, sklearn) are imported inside the function
    body rather than at module top, so that other symbols added to this
    module in Task 7 remain importable without torch installed.
    """
    import torch
    from torch.amp import autocast
    from sklearn.metrics import f1_score, accuracy_score

    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            cmask = batch['chunk_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            with autocast('cuda', enabled=use_fp16):
                logits = model(ids, mask, cmask)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, -1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if not all_labels:
        raise RuntimeError(
            "evaluate_model: loader produced no batches — the dataset is empty "
            "(did chunk removal skip every document?).")
    n = max(len(loader), 1)
    return (total_loss / n,
            f1_score(all_labels, all_preds, average='macro'),
            accuracy_score(all_labels, all_preds),
            all_preds, all_labels)
