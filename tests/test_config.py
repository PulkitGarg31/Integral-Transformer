import importlib.util
import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch not installed locally")


def test_config_defaults_match_legacy():
    from src.config import Config
    c = Config()
    assert c.pretrained_model == "law-ai/InLegalBERT"
    assert c.max_chunks == 24
    assert c.batch_size == 28
    # 10/3 (paper Table I): the 15/5 leak-free run peaked at epoch 3-5 for
    # every model and never improved afterwards, so the headroom was unused.
    assert c.num_epochs == 10
    assert c.label_smoothing == 0.1
    assert c.early_stopping_patience == 3
    assert c.dataset_path == "/workspace/pdfs/Supreme_Court_of_India"
    assert c.delhi_path == "/workspace/pdfs/Delhi_High_Court"


def test_set_seed_returns_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # seed_log.txt written to cwd
    from src.config import set_seed
    assert set_seed(123) == 123
    assert (tmp_path / "seed_log.txt").exists()


def test_set_seed_restores_legacy_speed_flags(tmp_path, monkeypatch):
    """legacy/auto.py:27-29 enabled cudnn autotune + TF32 at startup; the
    restructured pipeline must actually set them, not just claim so in a comment."""
    import torch
    monkeypatch.chdir(tmp_path)
    from src.config import set_seed
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    set_seed(7)
    assert torch.backends.cudnn.benchmark is True
    assert torch.get_float32_matmul_precision() == 'medium'
