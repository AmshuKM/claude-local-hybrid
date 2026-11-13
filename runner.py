"""Run a plan of small, checked coding tasks through a local model (aider + Ollama).

    cd <your project>                 # a git repo with no uncommitted changes
    python runner.py plan.json [--retries 2] [--dry-run]

plan.json:
    {"model": "qwen3.5:9b",                              optional
     "tasks": [{"id": "t1",
                "task": "what to change, precisely",
                "files": ["src/a.py"],                   files the model may edit
                "read": ["docs/spec.md"],                optional read-only context
                "check": "pytest -q tests/test_a.py",    exit code 0 means done
                "needs": ["t0"]}]}                       optional earlier task ids

Work happens on a new branch local-run/<timestamp>; your branch is never touched.
Each task gets a fresh model context. A failed attempt is rolled back and retried
with the check's output; a task that keeps failing is marked failed and every task
that needs it is skipped. Results go to <plan>.log.json after every task.

The check commands run in your shell exactly as written: only run plans you trust.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "qwen3.5:9b"
# Pinned: aider otherwise sizes the context per request, and Ollama reloads the model
# (about 19 s on a GTX 1080 Ti) every time the size changes.
NUM_CTX = 16384
AIDER_TIMEOUT, CHECK_TIMEOUT = 1200, 600  # seconds
TAIL = 2000  # characters of check output kept for the retry prompt and the log

ENV = {"OLLAMA_API_BASE": "http://127.0.0.1:11434", **os.environ, "PYTHONIOENCODING": "utf-8"}


def validate(plan):
    tasks = plan.get("tasks") if isinstance(plan, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("plan needs a non-empty 'tasks' list")
    seen = set()
    for t in tasks:
        for key in ("id", "task", "files", "check"):
            if not t.get(key):
                raise ValueError(f"task {t.get('id', '?')!r}: missing '{key}'")
        if t["id"] in seen:
            raise ValueError(f"duplicate task id {t['id']!r}")
        for n in t.get("needs", []):
            if n not in seen:
                raise ValueError(f"task {t['id']!r}: needs {n!r}, which is not an earlier task")
        seen.add(t["id"])
    return tasks


def sh(cmd, timeout=60, shell=False):
    try:
        p = subprocess.run(cmd, shell=shell, env=ENV, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout}s"


def git(*args):
    rc, out = sh(["git", *args])
    if rc:
        sys.exit(f"git {' '.join(args)} failed:\n{out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Run a plan of checked tasks through a local model.")
    ap.add_argument("plan", type=Path)
    ap.add_argument("--retries", type=int, default=2, help="extra attempts per task (default 2)")
    ap.add_argument("--dry-run", action="store_true", help="validate the plan and list the tasks")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        tasks = validate(plan)
    except ValueError as e:
        sys.exit(f"bad plan: {e}")
    model = plan.get("model", DEFAULT_MODEL)

    if args.dry_run:
        for t in tasks:
            print(f"{t['id']:12} needs={t.get('needs', [])} files={t['files']} check={t['check']}")
        print(f"{len(tasks)} tasks, model {model}: plan is valid")
        return

    if not shutil.which("aider"):
        sys.exit("aider is not on PATH (install: uv tool install --python 3.12 aider-chat)")
    git("rev-parse", "--is-inside-work-tree")
    if sh(["git", "config", "user.name"])[0] or sh(["git", "config", "user.email"])[0]:
        sys.exit("git user.name / user.email are not set, so commits would fail")
    settings = Path(tempfile.gettempdir()) / "runner-aider-settings.yml"
    settings.write_text(f"- name: ollama_chat/{model}\n  extra_params:\n    num_ctx: {NUM_CTX}\n")
    # failed attempts are rolled back with reset --hard, so uncommitted work would be lost
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("uncommitted changes in tracked files: commit or stash them first")

    base = git("rev-parse", "HEAD")
    branch = f"local-run/{datetime.now():%Y%m%d-%H%M%S}"
    git("switch", "-c", branch)

    log_path = args.plan.with_suffix(".log.json")
    log = {"plan": str(args.plan), "model": model, "branch": branch, "base": base,
           "started": datetime.now().isoformat(timespec="seconds"), "tasks": []}
    status = {}

    for t in tasks:
        tid, entry = t["id"], {"id": t["id"], "attempts": 0, "seconds": 0}
        blockers = [n for n in t.get("needs", []) if status[n] != "passed"]
        if blockers:
            entry.update(status="skipped", output=f"needs {blockers}, which did not pass")
        else:
            start, t0, feedback = git("rev-parse", "HEAD"), time.time(), ""
            entry["status"] = "failed"
            existed = {f for f in t["files"] if Path(f).exists()}
            for attempt in range(1, args.retries + 2):
                entry["attempts"] = attempt
                msg = t["task"] + feedback
                # --no-auto-commits: aider's commit message costs a second model call per task
                cmd = ["aider", "--model", f"ollama_chat/{model}", "--message", msg,
                       "--model-settings-file", str(settings), "--no-auto-commits",
                       "--yes-always", "--no-stream", "--no-pretty", "--no-check-update",
                       "--no-show-model-warnings", "--no-gitignore", "--map-tokens", "1024"]
                for r in t.get("read", []):
                    cmd += ["--read", r]
                _, aider_out = sh(cmd + t["files"], AIDER_TIMEOUT)
                rc, out = sh(t["check"], CHECK_TIMEOUT, shell=True)
                entry["output"] = out[-TAIL:]
                if rc == 0:
                    # shortcut: commits tracked edits plus the task's own files only; a new
                    # file the model creates outside "files" stays untracked. List it in the plan.
                    made = [f for f in t["files"] if Path(f).exists()]
                    if made:
                        git("add", "--", *made)
                    if git("status", "--porcelain", "--untracked-files=no"):
                        git("commit", "-qam", f"{tid}: {t['task'].splitlines()[0][:60]}")
                    entry.update(status="passed", commit=git("rev-parse", "--short", "HEAD"))
                    break
                entry["aider"] = aider_out[-TAIL:]
                git("reset", "--hard", start)
                for f in t["files"]:  # reset leaves files the model newly created
                    if f not in existed and Path(f).exists():
                        Path(f).unlink()
                feedback = (f"\n\nA previous attempt failed this check (`{t['check']}`) "
                            f"with:\n{out[-TAIL:]}")
            entry["seconds"] = round(time.time() - t0)
        status[tid] = entry["status"]
        log["tasks"].append(entry)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")  # crash-safe: every task
        print(f"{tid:12} {entry['status']:8} attempts={entry['attempts']} {entry['seconds']}s",
              flush=True)

    passed = sum(s == "passed" for s in status.values())
    print(f"\n{passed}/{len(tasks)} passed on branch {branch}\nlog: {log_path}\n"
          f"review:  git log {base[:7]}..{branch}\n"
          f"discard: git switch - && git branch -D {branch}")
    sys.exit(0 if passed == len(tasks) else 1)


if __name__ == "__main__":
    main()
