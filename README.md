# Integral Transformer

Architectural Optimization of Transformers for Long Legal Documents.

The **Integral Transformer** is a hybrid long-document architecture for Indian court
judgments. Each judgment is split into chunks, encoded with
[InLegalBERT](https://huggingface.co/law-ai/InLegalBERT), and the chunk representations
are fused through two parallel pathways: standard multi-head self-attention and a
multi-scale Radial Basis Function (RBF) integration kernel that aggregates evidence
across all chunk pairs without softmax normalisation. A learned gate balances the two
pathways at every layer. The first application is appeal-outcome classification
(allowed vs. dismissed) on Supreme Court of India judgments, with zero-shot transfer to
the Delhi High Court.

Capstone Project CPG-89, Computer Science and Engineering Department,
Thapar Institute of Engineering and Technology, Patiala.
Mentor: Dr. Shivani Sharma.

**Team:** Arvin Saini (102303049) · Lavish Gupta (102317132) · Lovleen Shukla (102315011) · Pulkit Garg (102317214)

## Results

All models are trained on 40,694 Supreme Court judgments under a **verdict-hidden**
protocol: the last chunk of every training document (where the court states the
outcome) is removed, so the models must learn from the reasoning rather than from
phrases such as "appeal dismissed". The same checkpoints are then evaluated on
53,352 Delhi High Court judgments without any retraining. Seed 16911, batch 28,
FP16, early stopping on validation macro-F1.

**Supreme Court test set, verdict hidden (in-domain)**

| Model | Loss | Macro-F1 | Accuracy |
|---|---|---|---|
| Truncation (first 512 tokens) | 0.6385 | 0.6308 | 0.6335 |
| Mean Pool | 0.4628 | 0.7818 | 0.7824 |
| Attention-Only | 0.5035 | 0.7874 | 0.7879 |
| **Integral Transformer** | **0.4529** | **0.7896** | **0.7900** |

**Delhi High Court, full documents (cross-dataset, no retraining)**

| Model | Loss | Macro-F1 | Accuracy |
|---|---|---|---|
| Truncation | 0.6227 | 0.5238 | 0.6596 |
| Mean Pool | 0.3226 | 0.8501 | 0.8777 |
| Attention-Only | 0.2256 | 0.9049 | 0.9261 |
| **Integral Transformer** | **0.2147** | **0.9151** | **0.9344** |

**Delhi High Court, verdict hidden (cross-dataset, no retraining)**

| Model | Loss | Macro-F1 | Accuracy |
|---|---|---|---|
| Truncation | 0.6276 | 0.5207 | 0.6557 |
| Mean Pool | 0.5942 | 0.6604 | 0.7079 |
| Attention-Only | 0.6758 | 0.6617 | 0.7043 |
| **Integral Transformer** | **0.5875** | **0.6666** | **0.7138** |

**Ablation (Delhi High Court full documents, one component removed at a time)**

| Variant | Macro-F1 | Accuracy | Δ F1 |
|---|---|---|---|
| Full Integral Transformer | **0.9151** | **0.9344** | +0.0000 |
| No adaptive gate | 0.9108 | 0.9309 | −0.0043 |
| Single-scale kernel | 0.9098 | 0.9284 | −0.0053 |
| No position bias | 0.9033 | 0.9244 | −0.0118 |
| No label smoothing | 0.9143 | 0.9339 | −0.0008 |

**Cost.** The integration pathway adds 4.4% parameters (129.79M vs 124.27M) and
about 1% wall-clock per epoch over the attention-only model (17.7 vs 17.6 min/epoch
on one H100 MIG 3g.40gb slice, batch 28, FP16; measured with
`benchmark_epoch_time.py`).

## Setup

```bash
python -m venv env && source env/bin/activate
# Install torch first, matching your CUDA driver. transformers>=4.57 needs torch>=2.6.
pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The pipeline reads the judgments as PDFs. Point it at the two corpora with
`--sc-path` and `--delhi-path` (defaults are in `src/config.py`).

## Quick start

```bash
python -m pytest tests/ -v          # 60 tests; torch-dependent ones skip without torch
python main.py --stage smoke        # 2-minute architecture check
python main.py                      # full pipeline: data -> train -> eval-sc -> eval-delhi
```

Every run writes its console output to `logs/`, per-epoch history to `history/`,
confusion matrices and comparison charts to `plots/`, and checkpoints to
`checkpoints/`. Training saves a checkpoint after every epoch and can be resumed with
`python main.py --stage train --resume`.

### Useful variants

```bash
python main.py --stage data                    # extract + tokenize + cache only
python main.py --stage train                   # train the 4 models
python main.py --stage eval-sc                 # Supreme Court test table
python main.py --stage eval-delhi              # Delhi tables (full, hidden n=1, hidden n=2)
python main.py --train-mode full               # full-document regime (default: hidden)
python main.py --n-remove 2                    # hide the last 2 chunks during training
python main.py --seed 42                       # different split / init
python ablation.py                             # retrain the four ablation variants
python ablation_delhi.py                       # evaluate them on Delhi High Court
python benchmark_epoch_time.py --repeats 3     # per-model epoch timing
```

## Method

1. **Chunking** (`src/preprocessing.py`): each judgment is segmented into
   non-overlapping 400-word windows; at most 24 chunks are kept (the first two, the
   last five, and a uniform sample of the middle), each tokenized to 512 tokens.
   A mask records which slots hold real content.
2. **Chunk encoding** (`src/models.py`): every chunk is encoded independently with
   InLegalBERT and mean-pooled; learnable chunk-position embeddings are added.
3. **Dual-pathway blocks** (`src/models.py`): two stacked blocks, each combining
   multi-head self-attention with a 4-head multi-scale RBF integration kernel
   (with learned position bias) through an adaptive gate, followed by masked
   max-pooling and a linear classifier.
4. **Training** (`src/training.py`): label-smoothed cross-entropy, AdamW with a
   higher learning rate for the integration parameters, mixed precision, early
   stopping on validation macro-F1.
5. **Baselines**: Attention-Only (same blocks without the kernel), Mean Pool
   (no cross-chunk block), and Truncation (first 512 tokens only). All four share
   the same chunk encoder and masking so the comparison is like for like.

## Code layout

| File | Responsibility |
|---|---|
| `main.py` | CLI and stage orchestration |
| `src/config.py` | hyperparameters, dataset paths, seed logging |
| `src/constants.py` | model roster (`MODEL_NAMES`) |
| `src/preprocessing.py` | PDF to text, outcome labelling, chunking |
| `src/dataset.py` | tokenization, fingerprinted caching, verdict hiding |
| `src/models.py` | Integral Transformer and the three baselines |
| `src/training.py` | shared training loop, checkpoints, early stopping, resume |
| `src/evaluation.py` | metrics, comparison tables, plots |
| `ablation.py`, `ablation_delhi.py` | ablation training and cross-dataset evaluation |
| `benchmark_epoch_time.py`, `plot_epoch_time.py` | epoch-time benchmark and figure |
| `tests/` | 60 unit tests |

## Project status

- [x] Objective 1: Integral Transformer architecture (RBF integration kernel, adaptive gate, InLegalBERT chunk encoder)
- [x] Objective 2: Supreme Court appeal-outcome classification with cross-jurisdiction evaluation on the Delhi High Court (results above)
- [ ] Objective 3: legal issue identification (multi-label area-of-law classification), in progress
- [ ] Objective 4: ratio decidendi extraction with headnote-paragraph weak supervision, in progress

Code for Objectives 3 and 4 will be added to this repository once the runs are complete.
