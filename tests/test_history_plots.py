import importlib.util
import json
import os

import pytest

matplotlib_available = importlib.util.find_spec("matplotlib") is not None


def test_history_path_format():
    from src.evaluation import history_path
    assert history_path("hidden1", "integral") == os.path.join(
        "history", "hidden1_integral_history.json")
    assert history_path("full", "baseline") == os.path.join(
        "history", "full_baseline_history.json")


def test_plot_training_curves_no_history_returns_none(tmp_path, monkeypatch, capsys):
    from src.evaluation import plot_training_curves
    monkeypatch.chdir(tmp_path)
    assert plot_training_curves("hidden1") is None
    assert "No training history" in capsys.readouterr().out


@pytest.mark.skipif(not matplotlib_available, reason="matplotlib not installed locally")
def test_plot_training_curves_writes_png(tmp_path, monkeypatch):
    from src.evaluation import plot_training_curves, history_path
    monkeypatch.chdir(tmp_path)
    os.makedirs("history")
    for name in ("integral", "baseline"):
        hist = {"train_loss": [0.6, 0.45], "val_loss": [0.5, 0.46],
                "val_f1": [0.80, 0.84], "val_acc": [0.81, 0.84]}
        with open(history_path("hidden1", name), "w", encoding="utf-8") as f:
            json.dump(hist, f)
    path = plot_training_curves("hidden1")
    assert path == os.path.join("plots", "training_curves_hidden1.png")
    assert os.path.exists(path)
