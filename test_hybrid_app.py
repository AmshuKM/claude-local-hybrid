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


def test_signed_out_reply_says_connect(tmp_path, monkeypatch):
    # a signed-out claude CLI prints valid JSON with is_error set, not garbage
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    h.prepare_project(str(tmp_path))
    real_run = h.run
    out = '{"is_error": true, "result": "Failed to authenticate", "session_id": "dead"}'
    monkeypatch.setattr(h, "run", lambda cmd, *a, **k:
                        (1, out, "") if cmd[0] == "claude" else real_run(cmd, *a, **k))
    monkeypatch.setattr(h, "claude_status", lambda: {"installed": True, "connected": False, "plan": None})
    monkeypatch.setattr(h, "PLANS", tmp_path / "plans")
    monkeypatch.setattr(h, "save", lambda: None)
    h.S["chats"]["c1"] = {"id": "c1", "title": "t", "project": str(tmp_path),
                          "session": None, "busy": "Claude", "messages": []}
    h.claude_turn("c1", ("hi", False))
    chat = h.S["chats"].pop("c1")
    assert "Click Connect" in chat["messages"][-1]["text"]
    assert chat["session"] is None and chat["busy"] is False
