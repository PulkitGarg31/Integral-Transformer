import io
import os
import subprocess
import sys
from pathlib import Path

OBJ1 = Path(__file__).resolve().parent.parent


def test_logging_survives_legacy_windows_console(tmp_path):
    """A cp1252 console must not crash on the emoji progress messages —
    setup_logging has to degrade the characters, not die before stage 1."""
    code = (f"import sys; sys.path.insert(0, {str(OBJ1)!r}); "
            "import main; main.setup_logging('smoke', 'hidden'); print('done')")
    env = dict(os.environ)
    env.pop('PYTHONIOENCODING', None)
    env.pop('PYTHONUTF8', None)
    r = subprocess.run([sys.executable, '-X', 'utf8=0', '-c', code],
                       cwd=tmp_path, capture_output=True, text=True,
                       timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    assert 'done' in r.stdout


def test_tee_writes_to_both_stream_and_file(tmp_path):
    from main import Tee
    real = io.StringIO()
    logpath = tmp_path / "out.log"
    with open(logpath, "w", encoding="utf-8") as f:
        tee = Tee(real, f)
        tee.write("hello 📝\n")
        tee.write("world\n")
        tee.flush()
    assert real.getvalue() == "hello 📝\nworld\n"
    assert logpath.read_text(encoding="utf-8") == "hello 📝\nworld\n"


def test_tee_delegates_stream_protocol(tmp_path):
    """Libraries probe sys.stdout for the full stream protocol (isatty,
    writable, encoding, ...) — Tee must delegate what it doesn't wrap."""
    from main import Tee
    real = io.StringIO()
    with open(tmp_path / "o.log", "w", encoding="utf-8") as f:
        tee = Tee(real, f)
        assert tee.isatty() is False
        assert tee.writable() is True


def test_setup_logging_creates_timestamped_file(tmp_path, monkeypatch):
    import sys
    import main as m
    monkeypatch.chdir(tmp_path)
    old_stdout = sys.stdout
    try:
        path = m.setup_logging("train", "hidden")
        print("captured line")
    finally:
        sys.stdout = old_stdout
    assert os.path.exists(path)
    name = os.path.basename(path)
    assert name.startswith("run_train_hidden_") and name.endswith(".log")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "captured line" in content
    assert "saved to" in content  # the announcement line itself is logged
