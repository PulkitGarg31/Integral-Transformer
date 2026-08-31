import importlib.util
import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch not installed locally")


def test_ckpt_names():
    from src.training import ckpt_names
    best, latest = ckpt_names('hidden', 1, 'integral')
    assert best == "hidden1_integral_best.pt"
    assert latest == "hidden1_integral_latest.pt"
    best, _ = ckpt_names('full', 1, 'baseline')
    assert best == "full_baseline_best.pt"     # n_remove ignored in full mode


def test_ckpt_names_respect_save_dir(tmp_path):
    import os
    from src.training import ckpt_names
    best, latest = ckpt_names('hidden', 1, 'integral', save_dir=str(tmp_path))
    assert best == os.path.join(str(tmp_path), "hidden1_integral_best.pt")
    assert latest == os.path.join(str(tmp_path), "hidden1_integral_latest.pt")


def _toy_integral():
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.integration_layers = nn.ModuleList([nn.Linear(4, 4)])
            self.gates = nn.ModuleList([nn.Linear(4, 4)])
            self.integration_probe = nn.Linear(4, 4)   # name contains 'integration'
            self.classifier = nn.Linear(4, 2)

    return Toy()


def test_build_optimizer_integral_has_three_groups():
    from src.training import build_optimizer
    from src.config import Config

    opt = build_optimizer(_toy_integral(), 'integral', Config())
    assert len(opt.param_groups) == 3
    assert opt.param_groups[1]['lr'] == Config.integration_lr

    opt2 = build_optimizer(_toy_integral(), 'baseline', Config())
    assert len(opt2.param_groups) == 1


def test_build_optimizer_groups_by_module_not_name_substring():
    """A module whose NAME merely contains 'integration' must not silently get
    the integration learning rate — grouping must be by module reference."""
    from src.training import build_optimizer
    from src.config import Config

    model = _toy_integral()
    opt = build_optimizer(model, 'integral', Config())
    integ_group = {id(p) for p in opt.param_groups[1]['params']}
    expected = {id(p) for p in model.integration_layers.parameters()}
    assert integ_group == expected
    probe = {id(p) for p in model.integration_probe.parameters()}
    other_group = {id(p) for p in opt.param_groups[2]['params']}
    assert probe <= other_group          # probe params ride at the base LR


# ---------------------------------------------------------------------------
# train_model behavior: skip / resume / checkpoint contents
# ---------------------------------------------------------------------------

def _tiny_model():
    import torch
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, ids, mask, cmask):
            feat = torch.ones(ids.shape[0], 4)
            return self.lin(feat)

    return Tiny()


def _tiny_batches():
    import torch
    return [{
        'input_ids': torch.ones(2, 3, 4, dtype=torch.long),
        'attention_mask': torch.ones(2, 3, 4, dtype=torch.long),
        'chunk_mask': torch.ones(2, 3),
        'labels': torch.tensor([0, 1]),
    }]


def _tiny_config(tmp_path, epochs):
    from src.config import Config
    config = Config()
    config.device = 'cpu'
    config.use_fp16 = False
    config.num_epochs = epochs
    config.gradient_accumulation_steps = 1
    config.early_stopping_patience = 999
    config.save_dir = str(tmp_path)
    config.resume_checkpoint = False
    return config


def _fake_eval_sequence(monkeypatch, f1s):
    calls = {'n': 0}

    def fake_eval(model, loader, device, use_fp16, criterion):
        f1 = f1s[min(calls['n'], len(f1s) - 1)]
        calls['n'] += 1
        return 0.5, f1, f1, [0, 1], [0, 1]

    monkeypatch.setattr('src.training.evaluate_model', fake_eval)


def test_resume_continues_interrupted_training(tmp_path, monkeypatch):
    import torch
    from src.training import train_model, ckpt_names
    monkeypatch.chdir(tmp_path)
    _fake_eval_sequence(monkeypatch, [0.5, 0.6, 0.7])
    loaders = _tiny_batches()

    config = _tiny_config(tmp_path, epochs=1)      # "interrupted" after epoch 1
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)
    _, latest = ckpt_names('hidden', 1, 'tiny', save_dir=str(tmp_path))
    assert torch.load(latest, weights_only=False)['epoch'] == 0

    config = _tiny_config(tmp_path, epochs=3)
    config.resume_checkpoint = True                # rerun with --resume
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)
    assert torch.load(latest, weights_only=False)['epoch'] == 2   # epochs 2-3 actually ran


def test_without_resume_existing_best_still_skips(tmp_path, monkeypatch):
    import torch
    from src.training import train_model, ckpt_names
    monkeypatch.chdir(tmp_path)
    _fake_eval_sequence(monkeypatch, [0.5])
    loaders = _tiny_batches()

    config = _tiny_config(tmp_path, epochs=1)
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)
    _, latest = ckpt_names('hidden', 1, 'tiny', save_dir=str(tmp_path))

    config = _tiny_config(tmp_path, epochs=3)      # no --resume: must skip
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)
    assert torch.load(latest, weights_only=False)['epoch'] == 0


def test_resume_skips_when_early_stopping_already_triggered(tmp_path, monkeypatch):
    import torch
    from src.training import train_model, ckpt_names
    monkeypatch.chdir(tmp_path)
    _fake_eval_sequence(monkeypatch, [0.5, 0.4])   # epoch 2 triggers patience=1
    loaders = _tiny_batches()

    config = _tiny_config(tmp_path, epochs=2)
    config.early_stopping_patience = 1
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)
    _, latest = ckpt_names('hidden', 1, 'tiny', save_dir=str(tmp_path))
    assert torch.load(latest, weights_only=False)['epoch'] == 1

    config = _tiny_config(tmp_path, epochs=5)      # resume must NOT train more:
    config.early_stopping_patience = 1             # early stopping already fired
    config.resume_checkpoint = True
    _fake_eval_sequence(monkeypatch, [0.9])
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)
    assert torch.load(latest, weights_only=False)['epoch'] == 1


def test_resume_reports_when_training_already_completed(tmp_path, monkeypatch, capsys):
    from src.training import train_model
    monkeypatch.chdir(tmp_path)
    _fake_eval_sequence(monkeypatch, [0.5, 0.6])
    loaders = _tiny_batches()

    config = _tiny_config(tmp_path, epochs=2)
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)

    config = _tiny_config(tmp_path, epochs=2)      # all epochs already done
    config.resume_checkpoint = True
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)
    assert "already completed" in capsys.readouterr().out


def test_checkpoint_records_current_best_f1_and_patience(tmp_path, monkeypatch):
    import torch
    from src.training import train_model, ckpt_names
    monkeypatch.chdir(tmp_path)
    _fake_eval_sequence(monkeypatch, [0.5, 0.8])   # improves on the final epoch
    loaders = _tiny_batches()

    config = _tiny_config(tmp_path, epochs=2)
    train_model(_tiny_model(), 'tiny', config, loaders, loaders, 'hidden', 1)
    best, latest = ckpt_names('hidden', 1, 'tiny', save_dir=str(tmp_path))
    for path in (best, latest):
        ck = torch.load(path, weights_only=False)
        assert ck['best_val_f1'] == 0.8            # NOT the stale pre-update value
        assert ck['no_improve'] == 0               # patience counter persisted
