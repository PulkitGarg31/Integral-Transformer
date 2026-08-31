import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

OBJ1 = Path(__file__).resolve().parent.parent

_torch_ok = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _torch_ok, reason="torch not installed locally")


def test_help_lists_all_stages():
    r = subprocess.run([sys.executable, "main.py", "--help"],
                       cwd=OBJ1, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    for token in ["--stage", "--train-mode", "--n-remove", "--resume",
                  "--sc-path", "--delhi-path", "smoke"]:
        assert token in r.stdout


class _EmptySource:
    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError(i)


@needs_torch
def test_stage_eval_sc_skips_empty_dataset(tmp_path, monkeypatch, capsys):
    """--n-remove >= every document's chunk count must skip gracefully,
    like stage_eval_delhi does, instead of crashing inside f1_score."""
    sys.path.insert(0, str(OBJ1))
    import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, 'load_or_build_sc_datasets',
                        lambda config, args: (None, None, _EmptySource()))
    from src.config import Config
    config = Config(); config.device = 'cpu'; config.save_dir = str(tmp_path)
    args = argparse.Namespace(train_mode='hidden', n_remove=24, seed=1)
    main.stage_eval_sc(config, args)                 # must not raise
    out = capsys.readouterr().out
    assert "no documents left" in out.lower()


@needs_torch
def test_sc_datasets_built_once_per_process(tmp_path, monkeypatch):
    """--stage all must not re-read the multi-GB cache from disk per stage."""
    sys.path.insert(0, str(OBJ1))
    import torch
    import main
    from src.config import Config
    from src.preprocessing import DataController

    monkeypatch.setattr(main, '_SC_DATASETS', None, raising=False)
    monkeypatch.setattr(main, 'require_path', lambda *a: None)
    monkeypatch.setattr(DataController, 'discover_pdfs',
                        lambda self, root: ['a.pdf', 'b.pdf', 'c.pdf'])
    monkeypatch.setattr(main.os.path, 'exists', lambda p: True)
    loads = {'n': 0}

    def fake_load(path, **kw):
        loads['n'] += 1
        return {'train': 'tr', 'val': 'va', 'test': 'te'}

    monkeypatch.setattr(torch, 'load', fake_load)
    config = Config(); config.device = 'cpu'
    args = argparse.Namespace(seed=1)
    r1 = main.load_or_build_sc_datasets(config, args)
    r2 = main.load_or_build_sc_datasets(config, args)
    assert r1 == r2 == ('tr', 'va', 'te')
    assert loads['n'] == 1
