"""uv run --python 3.12 --with pytest pytest -q test_hybrid_app.py"""
import subprocess

import hybrid_app as h


def test_recommend_fits_gpu_memory():
    assert h.recommend(11) == "qwen3.5:9b"
    assert h.recommend(8) == "qwen3.5:9b"
    assert h.recommend(6) == "qwen3.5:4b"
    assert h.recommend(4) == "qwen3.5:2b"
    assert h.recommend(None) == "qwen3.5:2b"  # no NVIDIA GPU


def test_chosen_model_falls_back_then_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "SETTINGS", tmp_path / "settings.json")
    assert h.chosen_model() == h.runner.DEFAULT_MODEL
    h.save_model("qwen3.5:4b")
    assert h.chosen_model() == "qwen3.5:4b"


def test_folder_inside_a_repo_gets_its_own(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    inner = tmp_path / "inner"
    inner.mkdir()
    h.prepare_project(str(inner))
    assert (inner / ".git").exists()
