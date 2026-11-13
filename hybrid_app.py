"""Hybrid app: chat with Claude to plan work, run the plans on the local model.

    uv run --python 3.12 hybrid_app.py        then open http://127.0.0.1:8765

Claude (via the `claude` CLI and your existing login) writes the tests and a plan;
runner.py executes the plan on the local Ollama model; results show in the chat.
The server only listens on 127.0.0.1 and rejects requests from other web origins,
because it can edit files and run commands in the projects you pick.
"""
import base64, json, os, platform, re, shutil, signal, subprocess, sys, threading, time
import urllib.request, uuid, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import runner

HERE = Path(__file__).parent
DATA = HERE / "hybrid-data"
PLANS, UPLOADS, CHATS = DATA / "plans", DATA / "uploads", DATA / "chats.json"
SETTINGS = DATA / "settings.json"  # {"model": ...}, written by the first-run setup
OLLAMA = "http://127.0.0.1:11434"
# Offered in the first-run setup, best first: (model, download GB, min GPU memory GB, plain-words note)
CHOICES = [
    ("qwen3.5:9b", 6.6, 8, "The best fit for graphics cards with 8 GB or more. Fast and reliable at small coding tasks."),
    ("qwen3.5:4b", 3.4, 5, "For graphics cards with 5 to 8 GB. Smaller, so it makes more mistakes."),
    ("qwen3.5:2b", 2.7, 0, "For small graphics cards or none. Fine for chat, weak at coding."),
]
PORT = 8765
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
CLAUDE_TIMEOUT = 1800  # seconds
MAX_UPLOAD = 25 * 1024 * 1024

LOCK = threading.RLock()
S = {"chats": {}}   # persisted to CHATS
RUN = {}            # the one active local run: proc, plan, project
CACHE = {}          # status / plugin list

SYSTEM = f"""You are the planning half of a hybrid coding app. A small local model (2 to 9B
parameters, running on the user's GPU through aider) makes the code edits. You plan, write
the checks, and review results. The user talks to you through a chat UI.

When a request needs code changes in this project:
1. Read what you need, then write or update the tests yourself. The local model must never
   write its own tests.
2. Write a plan as JSON to the exact plan path given at the end of the user's message:
   {{"model": "<the model named in the [app] line>", "tasks": [{{"id": "short-id", "task": "precise instructions",
   "files": ["file the model may edit"], "read": ["tests/test_x.py", "other context"],
   "check": "command that exits 0 only when the task is done", "needs": ["earlier-id"]}}]}}
   - Each task edits one or two files and is fully specified in its text: exact names,
     signatures, behaviour, edge cases. Put the test file in "read".
   - A task's check must be able to pass once that task and its "needs" are done. Never check
     with a test file that imports code a later task writes: in JS ES modules (and Python) one
     missing import fails the whole file, whatever test-name filter you add.
   - This machine runs {platform.system()}. For Python use checks like
     `uv run --python 3.12 --with pytest pytest -q tests/test_x.py`. For Node use
     `node --test tests/x.test.js` or a bare `node --test`; Node treats a directory argument
     such as `tests/` as a file and always fails.
   - Prefer more, smaller tasks. Known weaknesses of the local model: it cannot count
     characters (give limits in words), it repeats a mistake even when shown the error, it
     falls into name collisions (a parameter that hides an import), and it skips
     requirements buried in long text, so state critical requirements first and bluntly.
3. Do not write the implementation yourself, except a fix under about ten lines found in
   review. Never run the plan; the user presses Run.
4. Reply in plain text (the UI does not render markdown), two or three sentences on what
   the plan does. Do not list the tasks; the UI shows the plan as a card.

When asked to review a run, read the log JSON at the path given (per task: status,
attempts, output = check output, aider = the model's output). Work out why each failure
happened, fix what is yours to fix (a wrong test, unclear task wording), and write a
revised plan containing only the tasks that have not passed, to the new plan path.

If the request is a question or needs no code change, just answer and write no plan.
You have no shell here.

Every user message ends with an [app] line. Normally it gives the plan path. If it says
"Direct mode", the user has chosen to have you edit the files yourself for that message:
follow that line instead of the planning rules above."""


# Sent with each message, not as a system prompt: a resumed session keeps its original system
# prompt, so the mode has to travel with the message.
DIRECT_NOTE = ("[app] Direct mode: for this message the user chose to have you make the change "
               "yourself, overriding the planning rules. Edit the project files directly and add or "
               "update tests for what you change. Do not write a plan. You cannot run anything "
               "here, so say plainly what you did not run. Reply briefly in plain text: what you "
               "changed and what the user should check.")


def save():
    CHATS.write_text(json.dumps(S, indent=1), encoding="utf-8")


def run(cmd, cwd=None, timeout=60, input=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, input=input, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (subprocess.TimeoutExpired, OSError) as e:
        return -1, "", str(e)


def git(project, *args):
    rc, out, err = run(["git", *args], project)
    return rc, (out + err).strip()


def changed(project):
    """Paths git sees as modified or untracked."""
    _, out, _ = run(["git", "status", "--porcelain", "-uall", "-z"], project)
    paths, entries, i = set(), out.split("\0"), 0
    while i < len(entries):
        e = entries[i]
        if e:
            paths.add(e[3:])
            if e[0] in "RC":  # renames and copies are followed by the original path
                i += 1
        i += 1
    return paths


def prepare_project(project):
    """The runner needs a git repo with an identity and at least one commit."""
    rc, top = git(project, "rev-parse", "--show-toplevel")
    # a folder inside another repo gets its own: runs must never branch or reset the outer one
    if rc or Path(top).resolve() != Path(project).resolve():
        git(project, "init")
    if git(project, "config", "user.name")[0]:
        git(project, "config", "user.name", "hybrid-app")
        git(project, "config", "user.email", "hybrid-app@localhost")
    if git(project, "rev-parse", "HEAD")[0]:
        git(project, "commit", "--allow-empty", "-m", "Start")


def claude_turn(cid, request):
    prompt, direct = request  # direct: Claude edits the project itself instead of planning
    with LOCK:
        chat = S["chats"][cid]
        project, session = chat["project"], chat.get("session")
        n = sum(m["role"] == "assistant" for m in chat["messages"])
    plan = PLANS / f"{cid}-{n}.json"
    before = changed(project)
    cmd = ["claude", "-p", "--output-format", "json", "--permission-mode", "acceptEdits",
           "--allowedTools", "Read,Glob,Grep,Write,Edit", "--add-dir", str(DATA),
           "--append-system-prompt", SYSTEM]
    if session:
        cmd += ["--resume", session]
    prompt += "\n\n" + (DIRECT_NOTE if direct
                        else f"[app] Plan path for this turn, if a plan is needed: {plan}"
                             f" Plan model: {chosen_model()}")
    rc, out, err = run(cmd, project, CLAUDE_TIMEOUT, prompt)
    try:
        reply = json.loads(out)
        text, session = reply.get("result") or "(no reply)", reply.get("session_id", session)
    except json.JSONDecodeError:
        CACHE["claude_at"] = 0  # re-check: a failed call is often a signed-out CLI
        text = ("Claude isn't connected. Click Connect at the bottom left, sign in, then send again."
                if not claude_status()["connected"] else f"The Claude call failed.\n{(err or out)[-800:]}")
    msg = {"role": "assistant", "text": text, **({"by": "claude"} if direct else {})}
    if plan.exists():
        try:
            runner.validate(json.loads(plan.read_text(encoding="utf-8")))
            msg["plan"] = str(plan)
        except (ValueError, json.JSONDecodeError) as e:
            msg["text"] += f"\n\n(The plan Claude wrote is not valid and cannot be run: {e})"
    new = sorted(changed(project) - before)
    if new:  # commit what Claude wrote so the runner starts from a clean tree
        git(project, "add", "--", *new)
        rc, out = git(project, "commit", "-m", f"{'Claude' if direct else 'Plan'}: {chat['title']}", "--", *new)
        if rc:
            msg["text"] += f"\n\n(Could not commit Claude's files: {out[-300:]})"
    with LOCK:
        chat["messages"].append(msg)
        chat.update(session=session, busy=False)
        save()


def local_turn(cid, files):
    """Offline chat: the local model answers directly. It has no tools and writes no plans."""
    with LOCK:
        chat = S["chats"][cid]
        history = [{"role": m["role"], "content": m["text"]} for m in chat["messages"][-20:]]
    images = []
    for f in files:
        p = Path(f["path"])
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            images.append(base64.b64encode(p.read_bytes()).decode())
        else:
            try:
                body = p.read_text(encoding="utf-8")[:20000]
            except UnicodeDecodeError:
                body = "(binary file, not shown)"
            history[-1]["content"] += f"\n\nAttached file {f['name']}:\n{body}"
    if images:
        history[-1]["images"] = images
    system = ("You are a coding assistant running offline on the user's own GPU. You cannot read "
              f"or change files; the user's project folder is {chat['project']}. Answer briefly "
              "in plain text, because the UI does not render markdown.")
    req = urllib.request.Request(OLLAMA + "/api/chat", json.dumps({
        "model": chosen_model(), "stream": False, "think": False,  # no thinking: faster chat
        "options": {"num_ctx": runner.NUM_CTX},  # same size as the runner, so no model reload
        "messages": [{"role": "system", "content": system}, *history]}).encode(),
        {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.load(r)["message"]["content"].strip() or "(no reply)"
    except (OSError, KeyError, json.JSONDecodeError) as e:
        text = f"The local model did not answer. Is Ollama running?\n{e}"
    with LOCK:
        chat["messages"].append({"role": "assistant", "text": text, "by": "local"})
        chat["busy"] = False
        save()


def guarded(turn, cid, arg):
    try:
        turn(cid, arg)
    except Exception as e:  # a crashed turn must not leave the chat stuck on "thinking"
        with LOCK:
            S["chats"][cid]["messages"].append({"role": "assistant", "text": f"Something went wrong: {e}"})
            S["chats"][cid]["busy"] = False
            save()


def plan_view(path):
    p = Path(path)
    tasks = json.loads(p.read_text(encoding="utf-8"))["tasks"]
    log = None
    try:
        log = json.loads(p.with_suffix(".log.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # no log yet, or caught mid-write
        pass
    done = {t["id"]: t for t in log["tasks"]} if log else {}
    running = RUN.get("plan") == str(p) and RUN["proc"].poll() is None
    rows, current = [], running
    for t in tasks:
        d = done.get(t["id"])
        status = d["status"] if d else "running" if current else "waiting"
        if not d:
            current = False  # only the first unfinished task is the running one
        rows.append({"id": t["id"], "summary": t["task"][:140], "status": status,
                     "attempts": d["attempts"] if d else 0, "seconds": d["seconds"] if d else 0})
    state = ("running" if running else "ready" if not log
             else "done" if len(done) == len(tasks) else "stopped")
    return {"path": str(p), "tasks": rows, "state": state, "branch": (log or {}).get("branch")}


def task_detail(plan, tid):
    log = json.loads(Path(plan).with_suffix(".log.json").read_text(encoding="utf-8"))
    t = next(x for x in log["tasks"] if x["id"] == tid)
    parts = [f"{t['status']} after {t['attempts']} attempt(s), {t['seconds']} s",
             "--- check output ---\n" + t.get("output", "")]
    if t.get("aider"):
        parts.append("--- local model output (last attempt) ---\n" + t["aider"])
    if t.get("commit"):
        project = next(c["project"] for c in S["chats"].values()
                       if any(m.get("plan") == plan for m in c["messages"]))
        parts.append("--- changes ---\n" + git(project, "show", "--stat", "-p", t["commit"])[1][:20000])
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n\n".join(parts))  # strip terminal colour codes


def status():
    if time.time() - CACHE.get("status_at", 0) > 5:
        _, ps, _ = run(["ollama", "ps"], timeout=10)
        loaded = [ln.split()[0] for ln in ps.splitlines()[1:] if ln.strip()]
        _, gpu, _ = run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                         "--format=csv,noheader,nounits"], timeout=10)
        m = re.match(r"\s*(\d+)\s*,\s*(\d+)", gpu)
        CACHE.update(status_at=time.time(), status={
            "model": ", ".join(loaded) or "no model loaded",
            "gpu": f"{int(m[1]) / 1024:.1f} of {int(m[2]) / 1024:.0f} GB" if m else "unknown"})
    return CACHE["status"]


def claude_status():
    """Is the claude CLI installed and signed in? Cached: the check takes about a second."""
    ttl = 3 if time.time() - CACHE.get("login_at", 0) < 300 else 30  # check often while signing in
    if time.time() - CACHE.get("claude_at", 0) > ttl:
        rc, out, _ = run(["claude", "auth", "status", "--json"], timeout=20)
        try:
            j = json.loads(out)
            st = {"installed": True, "connected": bool(j.get("loggedIn")),
                  "plan": j.get("subscriptionType") or j.get("authMethod")}
        except json.JSONDecodeError:
            st = {"installed": rc != -1, "connected": False, "plan": None}  # -1: claude not found
        CACHE.update(claude_at=time.time(), claude=st)
    return CACHE["claude"]


def claude_login():
    # a visible terminal window, because sign-in may ask the user to paste a code
    if os.name == "nt":
        args = ["cmd", "/c", "start", "Sign in to Claude", "cmd", "/k", "claude auth login"]
    elif sys.platform == "darwin":
        args = ["osascript", "-e", 'tell application "Terminal" to do script "claude auth login"',
                "-e", 'tell application "Terminal" to activate']
    else:
        term = next((t for t in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm")
                     if shutil.which(t)), None)
        if not term:
            raise OSError("no terminal app found. Open a terminal and run: claude auth login")
        inner = ["sh", "-c", 'claude auth login; exec "${SHELL:-sh}"']
        args = [term, "--", *inner] if term == "gnome-terminal" else [term, "-e", *inner]
    subprocess.Popen(args)
    CACHE.update(login_at=time.time(), claude_at=0)


PULL = {}  # the one model download in progress: model, done, total (bytes), error


def chosen_model():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))["model"]
    except (OSError, ValueError, KeyError):
        return runner.DEFAULT_MODEL


def recommend(vram_gb):
    """The best choice that fits in GPU memory; no NVIDIA GPU (None) gets the smallest."""
    return next(m for m, _, need, _ in CHOICES if (vram_gb or 0) >= need)


def ollama_models():
    """Installed model names, or None when Ollama is not running."""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5) as r:
            return [m["name"] for m in json.load(r)["models"]]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def setup_info():
    _, out, _ = run(["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"], timeout=10)
    m = re.match(r"\s*(.+?)\s*,\s*(\d+)", out)  # first GPU only
    gpu = {"name": m[1], "gb": round(int(m[2]) / 1024)} if m else None
    if not gpu and sys.platform == "darwin" and platform.machine() == "arm64":
        _, mem, _ = run(["sysctl", "-n", "hw.memsize"], timeout=10)
        if mem.strip().isdigit():
            # shortcut: macOS lets the GPU use about 2/3 of shared memory by default
            gpu = {"name": "Apple chip (shared memory)", "gb": round(int(mem) / 2**30 * 2 / 3)}
    installed = ollama_models()
    return {"done": SETTINGS.exists(), "model": chosen_model(), "ollama": installed is not None,
            "installed": installed or [], "gpu": gpu, "recommended": recommend(gpu and gpu["gb"]),
            "choices": [{"model": c[0], "gb": c[1], "note": c[3]} for c in CHOICES],
            "pull": dict(PULL)}


def save_model(model):
    SETTINGS.write_text(json.dumps({"model": model}), encoding="utf-8")


def pull(model):
    req = urllib.request.Request(OLLAMA + "/api/pull", json.dumps({"model": model}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                msg = json.loads(line)
                if "error" in msg:
                    raise OSError(msg["error"])
                # shortcut: shows the largest layer only (the weights, ~all of the size)
                if msg.get("total", 0) >= PULL.get("total", 0):
                    PULL.update(done=msg.get("completed", 0), total=msg.get("total", 0))
        save_model(model)
        PULL.clear()
    except (OSError, ValueError) as e:
        PULL.update(error=f"The download stopped: {e}. Check your internet connection and try again.")


def plugins():
    if "plugins" not in CACHE:
        _, out, _ = run(["claude", "plugin", "list", "--json"], timeout=60)
        try:
            CACHE["plugins"] = sorted(p["id"].split("@")[0] for p in json.loads(out) if p.get("enabled"))
        except (json.JSONDecodeError, KeyError, TypeError):
            CACHE["plugins"] = []
    return CACHE["plugins"]


def pick_folder():
    code = ("import tkinter as t, tkinter.filedialog as f; r = t.Tk(); r.withdraw(); "
            "r.attributes('-topmost', 1); print(f.askdirectory(title='Pick a project folder'))")
    return run([sys.executable, "-c", code], timeout=600)[1].strip()


def stop_run():
    proc = RUN.get("proc")
    if proc and proc.poll() is None:
        # kill the runner together with aider underneath it
        if os.name == "nt":
            run(["taskkill", "/T", "/F", "/PID", str(proc.pid)])
        else:
            os.killpg(proc.pid, signal.SIGKILL)  # the runner leads its own process group
        proc.wait()
        project = RUN["project"]
        if git(project, "branch", "--show-current")[1].startswith("local-run/"):
            # shortcut: drops the interrupted attempt's edits to tracked files; a file it
            # newly created stays behind untracked. Delete by hand if one ever matters.
            git(project, "reset", "--hard")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def trusted(self):
        # blocks other websites (CSRF) and DNS rebinding from driving this server
        origin = self.headers.get("Origin")
        ok = self.headers.get("Host") in ALLOWED_HOSTS and (
            origin is None or origin.split("//")[-1] in ALLOWED_HOSTS)
        if not ok:
            self.send({"error": "forbidden"}, 403)
        return ok

    def do_GET(self):
        if not self.trusted():
            return
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self.send((HERE / "hybrid_app.html").read_bytes(), ctype="text/html; charset=utf-8")
            elif url.path == "/api/state":
                with LOCK:
                    chat = S["chats"].get(q.get("chat"))
                    view = chat and {**chat, "messages": [
                        {**m, "plan": plan_view(m["plan"])} if m.get("plan") else m
                        for m in chat["messages"]]}
                    chats = [{"id": c["id"], "title": c["title"], "project": c["project"]}
                             for c in S["chats"].values()]
                running = bool(RUN) and RUN["proc"].poll() is None
                self.send({"chats": chats, "chat": view, "status": status(), "running": running,
                           "setup": SETTINGS.exists(), "model": chosen_model(),
                           "claude": claude_status()})
            elif url.path == "/api/setup":
                self.send(setup_info())
            elif url.path == "/api/plugins":
                self.send(plugins())
            elif url.path == "/api/task":
                self.send({"text": task_detail(self.own_plan(q["plan"]), q["id"])})
            else:
                self.send({"error": "not found"}, 404)
        except (KeyError, StopIteration, OSError, ValueError) as e:
            self.send({"error": f"bad request: {e}"}, 400)

    def own_plan(self, path):
        """Only plans this app stored may be run or inspected."""
        p = Path(path).resolve()
        if p.parent != PLANS.resolve() or not p.exists():
            raise ValueError("unknown plan")
        return str(p)

    def do_POST(self):
        if not self.trusted():
            return
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        length = int(self.headers.get("Content-Length") or 0)
        try:
            if url.path == "/api/upload":
                if length > MAX_UPLOAD:
                    return self.send({"error": "file is larger than 25 MB"}, 413)
                if q["chat"] not in S["chats"]:
                    raise KeyError("chat")
                name = re.sub(r"[^\w.\- ]", "_", Path(q["name"]).name) or "file"
                dest = UPLOADS / q["chat"] / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(self.rfile.read(length))
                return self.send({"name": name, "path": str(dest)})
            body = json.loads(self.rfile.read(length) or b"{}")
            if url.path == "/api/pick-folder":
                self.send({"path": pick_folder()})
            elif url.path == "/api/new":
                project = Path(body["project"])
                if not project.is_dir():
                    return self.send({"error": "that folder does not exist"}, 400)
                prepare_project(str(project))
                cid = uuid.uuid4().hex[:12]
                with LOCK:
                    S["chats"][cid] = {"id": cid, "title": "New chat", "project": str(project),
                                       "session": None, "busy": False, "messages": []}
                    save()
                self.send({"id": cid})
            elif url.path == "/api/chat":
                with LOCK:
                    chat = S["chats"][body["chat"]]
                    if chat["busy"]:
                        return self.send({"error": "Claude is still answering"}, 409)
                    files = [f for f in body.get("files", [])
                             if Path(f["path"]).resolve().parent == (UPLOADS / chat["id"]).resolve()]
                    # hybrid: Claude plans, the GPU codes. claude: Claude edits. local: offline chat.
                    mode = "hybrid" if body.get("fix") else body.get("mode", "hybrid")
                    if mode not in ("hybrid", "claude", "local"):
                        raise ValueError("unknown mode")
                    local = mode == "local"
                    chosen = [] if local else [p for p in body.get("plugins", []) if p in plugins()]
                    if body.get("fix"):
                        log = Path(self.own_plan(body["fix"])).with_suffix(".log.json")
                        text = "Fix the failures from the last run"
                        prompt = (f"The local run of the last plan has finished. Review it: the log "
                                  f"is at {log}. Then write a revised plan for what did not pass.")
                    else:
                        text = prompt = body["text"].strip()
                        if not text:
                            return self.send({"error": "Enter a message first"}, 400)
                    if files:
                        prompt += "\n\nAttached files (read them):\n" + "\n".join(f["path"] for f in files)
                    if chosen:
                        prompt += "\n\nUse these installed plugins for this request: " + ", ".join(chosen)
                    if not chat["messages"]:
                        chat["title"] = text[:40]
                    chat["messages"].append({"role": "user", "text": text, "plugins": chosen,
                                             "files": [f["name"] for f in files]})
                    chat["busy"] = "Local model" if local else "Claude"  # shown as "... is thinking"
                    save()
                args = ((local_turn, chat["id"], files) if local
                        else (claude_turn, chat["id"], (prompt, mode == "claude")))
                threading.Thread(target=guarded, args=args, daemon=True).start()
                self.send({"ok": True})
            elif url.path == "/api/run":
                plan = self.own_plan(body["plan"])
                with LOCK:
                    if RUN and RUN["proc"].poll() is None:
                        return self.send({"error": "another run is already using the GPU"}, 409)
                    project = S["chats"][body["chat"]]["project"]
                    out = open(Path(plan).with_suffix(".out.txt"), "w", encoding="utf-8")
                    RUN.update(plan=plan, project=project, proc=subprocess.Popen(
                        [sys.executable, str(HERE / "runner.py"), plan],
                        cwd=project, stdout=out, stderr=subprocess.STDOUT,
                        start_new_session=os.name != "nt"))  # lets Stop kill the whole group
                self.send({"ok": True})
            elif url.path == "/api/setup":
                model, installed = body["model"], ollama_models() or []
                if model not in installed + [c[0] for c in CHOICES]:
                    raise ValueError("unknown model")
                if model in installed:
                    save_model(model)
                elif not PULL.get("model") or PULL.get("error"):  # one download at a time
                    PULL.clear()
                    PULL.update(model=model, done=0, total=0)
                    threading.Thread(target=pull, args=(model,), daemon=True).start()
                self.send(setup_info())
            elif url.path == "/api/claude-login":
                claude_login()
                self.send({"ok": True})
            elif url.path == "/api/stop":
                stop_run()
                self.send({"ok": True})
            else:
                self.send({"error": "not found"}, 404)
        except (KeyError, ValueError, OSError) as e:
            self.send({"error": f"bad request: {e}"}, 400)


def main():
    for d in (PLANS, UPLOADS):
        d.mkdir(parents=True, exist_ok=True)
    if CHATS.exists():
        S.update(json.loads(CHATS.read_text(encoding="utf-8")))
        for c in S["chats"].values():
            c["busy"] = False  # a Claude call cannot survive a restart
    print(f"Hybrid app: http://127.0.0.1:{PORT}   (Ctrl+C to stop)")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
        if "--open" in sys.argv:  # the double-click launchers pass this
            webbrowser.open(f"http://127.0.0.1:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        stop_run()


if __name__ == "__main__":
    main()
