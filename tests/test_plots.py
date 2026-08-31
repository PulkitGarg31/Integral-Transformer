import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

plotting_available = (importlib.util.find_spec("matplotlib") is not None
                      and importlib.util.find_spec("sklearn") is not None)


def test_save_plots_empty_results_returns_nothing(tmp_path, monkeypatch):
    from src.evaluation import save_plots
    monkeypatch.chdir(tmp_path)
    assert save_plots({}, "sc_hidden1") == []
    assert not os.path.exists("plots")


@pytest.mark.skipif(not plotting_available, reason="matplotlib/sklearn not installed locally")
def test_save_plots_writes_confusion_and_comparison(tmp_path, monkeypatch):
    from src.evaluation import save_plots
    monkeypatch.chdir(tmp_path)
    results = {
        "integral": {"loss": 0.4, "f1": 0.83, "acc": 0.83,
                     "preds": [0, 1, 1, 0, 1], "labels": [0, 1, 0, 0, 1]},
        "baseline": {"loss": 0.5, "f1": 0.80, "acc": 0.80,
                     "preds": [0, 1, 0, 0, 1], "labels": [0, 1, 0, 0, 1]},
    }
    paths = save_plots(results, "sc_hidden1")
    assert len(paths) == 2
    for p in paths:
        assert os.path.exists(p) and p.endswith(".png")
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["comparison_sc_hidden1.png", "confusion_sc_hidden1.png"]


def test_plot_helpers_run_without_torch(tmp_path):
    """The plotting helpers must not pull torch in.

    src/models.py imports torch at module level, so taking MODEL_NAMES from
    there breaks src/evaluation.py's torch-free contract and makes these
    helpers unusable on a laptop. They import it from src/constants.py
    instead; this test blocks torch in a subprocess to keep it that way
    even on machines where torch IS installed.
    """
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = textwrap.dedent("""
        import sys
        from importlib.abc import MetaPathFinder

        class BlockTorch(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ImportError("torch is blocked in this test")
                return None

        sys.meta_path.insert(0, BlockTorch())
        from src.evaluation import plot_training_curves, save_plots

        assert plot_training_curves("no_such_prefix") is None
        save_plots({"integral": {"loss": 0.4, "f1": 0.8, "acc": 0.8,
                                 "preds": [0, 1], "labels": [0, 1]}}, "torchless")
        print("OK")
    """)
    # utf-8 decoding: the helpers print emoji, which the Windows default
    # console codec (cp1252) cannot decode
    env = dict(os.environ, PYTHONPATH=pkg_root, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path),
                          env=env, capture_output=True,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_save_plots_without_matplotlib_is_graceful(tmp_path, monkeypatch):
    if plotting_available:
        pytest.skip("matplotlib installed — graceful-degradation path not reachable")
    from src.evaluation import save_plots
    monkeypatch.chdir(tmp_path)
    results = {"integral": {"loss": 0.4, "f1": 0.83, "acc": 0.83,
                            "preds": [0, 1], "labels": [0, 1]}}
    assert save_plots(results, "sc_hidden1") == []   # no crash, just skips
