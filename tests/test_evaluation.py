import importlib.util

import pytest

_torch_ok = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _torch_ok, reason="torch not installed locally")


def _toy_model():
    import torch
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, ids, mask, cmask):
            logits = torch.zeros(ids.shape[0], 2)
            logits[:, 0] = 10.0                      # always predicts class 0
            return logits + 0.0 * self.lin(torch.ones(ids.shape[0], 4)).sum()

    return Toy()


def _single_class_loader():
    import torch
    return [{
        'input_ids': torch.ones(2, 3, 4, dtype=torch.long),
        'attention_mask': torch.ones(2, 3, 4, dtype=torch.long),
        'chunk_mask': torch.ones(2, 3),
        'labels': torch.tensor([0, 0]),              # single-class subset
    }]


def _eval_config(tmp_path):
    from src.config import Config
    config = Config()
    config.device = 'cpu'
    config.use_fp16 = False
    config.save_dir = str(tmp_path)
    return config


@needs_torch
def test_eval_single_class_subset_does_not_crash(tmp_path, monkeypatch):
    import torch
    import src.models
    from src.evaluation import evaluate_checkpoints
    from src.training import ckpt_names
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(src.models.MODEL_REGISTRY, 'toy', lambda cfg: _toy_model())

    best, _ = ckpt_names('full', 1, 'toy', save_dir=str(tmp_path))
    torch.save({'model_state_dict': _toy_model().state_dict()}, best)

    results = evaluate_checkpoints(['toy'], _eval_config(tmp_path),
                                   _single_class_loader(), 'full', 1, title="t")
    assert 'toy' in results                          # must not raise ValueError


@needs_torch
def test_eval_corrupt_checkpoint_skips_model_not_whole_stage(tmp_path, monkeypatch):
    import torch
    import src.models
    from src.evaluation import evaluate_checkpoints
    from src.training import ckpt_names
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(src.models.MODEL_REGISTRY, 'bad', lambda cfg: _toy_model())
    monkeypatch.setitem(src.models.MODEL_REGISTRY, 'good', lambda cfg: _toy_model())

    bad_best, _ = ckpt_names('full', 1, 'bad', save_dir=str(tmp_path))
    with open(bad_best, 'wb') as f:
        f.write(b"this is not a torch checkpoint")
    good_best, _ = ckpt_names('full', 1, 'good', save_dir=str(tmp_path))
    torch.save({'model_state_dict': _toy_model().state_dict()}, good_best)

    results = evaluate_checkpoints(['bad', 'good'], _eval_config(tmp_path),
                                   _single_class_loader(), 'full', 1, title="t")
    assert 'good' in results                         # healthy model still evaluated
    assert 'bad' not in results                      # corrupt one skipped, not fatal


@needs_torch
def test_evaluate_model_empty_loader_raises_clear_error():
    import torch.nn as nn
    from src.evaluation import evaluate_model
    with pytest.raises(RuntimeError, match="empty"):
        evaluate_model(_toy_model(), [], 'cpu', False, nn.CrossEntropyLoss())


def test_print_comparison_formats_table_and_improvements():
    from src.evaluation import print_comparison
    results = {
        'integral':  {'loss': 0.44, 'f1': 0.7379, 'acc': 0.7955},
        'baseline':  {'loss': 0.57, 'f1': 0.7210, 'acc': 0.7729},
        'truncation': {'loss': 0.72, 'f1': 0.5251, 'acc': 0.6139},
    }
    out = print_comparison(results, ['integral', 'baseline', 'truncation', 'meanpool'])
    assert "INTEGRAL" in out and "0.7379" in out
    assert "MEANPOOL" not in out          # model missing from results is skipped entirely
    assert "vs BASELINE" in out
