"""Configuration and reproducibility utilities for the Integral Legal Transformer."""

import datetime
import random

import numpy as np
import torch


def set_seed(seed=None):
    """
    Set global random seed for reproducibility.
    If seed is None, generates a random one and logs it.
    Seed is printed + saved to seed_log.txt so you can rerun any experiment.
    """
    if seed is None:
        seed = random.randint(0, 99_999)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Legacy startup speed flags (legacy/auto.py:27-29): cuDNN autotune + TF32
    # matmul. For 100% determinism set benchmark=False and precision='highest'.
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('medium')  # TF32 on Ampere+

    # Log to console
    print(f"\n🌱 Random Seed: {seed}  ← save this to reproduce exact results")

    # Append to seed_log.txt so you never lose it
    log_entry = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | seed={seed}\n"
    with open("seed_log.txt", "a") as f:
        f.write(log_entry)

    return seed


class Config:
    """Configuration for the Integral Transformer (values identical to legacy/auto.py)."""

    # Model Architecture
    pretrained_model = "law-ai/InLegalBERT"
    hidden_size = 768
    num_attention_heads = 8
    num_integration_heads = 4
    integration_kernel_size = 64
    num_classes = 2
    dropout = 0.1
    max_length = 512
    max_chunks = 24                       # 🚀 Increased to 24: favors Integral for very long documents

    # Integration Module Hyperparameters
    integration_temperature = 1.0
    use_global_integration = True
    integration_layers = 2
    integration_dropout = 0.1             # kernel-weight + output dropout in the RBF kernel
                                          # (parity with the attention pathway); 0.0 = v2 kernel

    # Training — 🚀 DGX NODE OPTIMIZED v2 (40GB VRAM, safely using ~28-30GB at batch=28)
    batch_size = 28                       # 🚀 Increased to 28 (safe middle ground for fast training without OOM risk)
    gradient_accumulation_steps = 1       # 🚀 v2: was 2  → DGX: was 4  (orig) — eff. batch=32, no waiting
    learning_rate = 2e-5
    integration_lr = 5e-4                 # 🔥 Higher LR for integration (needs to learn faster)
    num_epochs = 10                       # paper Table I; run 1 (15 epochs) peaked at epoch 3-5
                                          # for every model, so the extra epochs only overfit
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    label_smoothing = 0.1                 # 🔥 Prevents overconfident predictions
    use_fp16 = True                       # 🔥 ENABLED — ~2x faster
    early_stopping_patience = 3           # paper Table I; run 1 (patience 5) never improved after
                                          # its epoch-3-5 peak, so the extra patience was unused
    # Linear LR schedule kept intentionally — cosine keeps LR high mid-training,
    # which can cause val F1 dips and false-trigger early stopping with patience=3.

    # Data
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1

    # Paths
    save_dir = "./checkpoints"
    log_dir = "./logs"
    dataset_path = "/workspace/pdfs/Supreme_Court_of_India"
    delhi_path = "/workspace/pdfs/Delhi_High_Court"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training Resumption
    resume_checkpoint = False
