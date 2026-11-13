# Claude + local model hybrid

Plan coding work with Claude, then run it on a small local model on your own GPU.

Claude writes the tests and a plan of small tasks. A local Ollama model (through aider) makes
the edits, one task at a time, each in a fresh context. A task only counts as done when its
check command passes, so the local model never has to drive tools or judge its own work.

Built and tested on Windows with a GTX 1080 Ti (11 GB) and `qwen3.5:9b`.

## Requirements

- [Ollama](https://ollama.com), running (the app's first-run setup picks and downloads a model)
- [uv](https://docs.astral.sh/uv/) and git
- aider: `uv tool install --python 3.12 aider-chat`
- [Claude Code](https://claude.com/claude-code), signed in (used for planning; not needed to run plans)

## The app

Double-click **Start Hybrid.bat**, or run:

```
uv run --python 3.12 hybrid_app.py
```

Open http://127.0.0.1:8765. On first start the app checks your graphics card and recommends a
local model (downloading it if needed); change it later with **Change** next to the model name.
Then start a chat, pick a project folder and describe what you want.

- Claude replies with a plan card. Press **Run on your GPU** to execute it locally.
- Click a task to see its check output and code changes.
- If tasks fail, **Ask Claude to fix failures** sends the log back for a revised plan.
- The paperclip attaches files and photos, or tells Claude to use one of your plugins.
- The **Hybrid / Claude / Local** switch picks who does the work for each message:
  - **Hybrid** (default): Claude plans and writes the tests, your GPU writes the code.
  - **Claude**: Claude makes the change itself, directly in the project.
  - **Local**: the local model answers. That works offline, but it is chat only: the
    local model cannot read your project or make plans.

Work happens on a new `local-run/<timestamp>` branch in the project; a failed attempt is
rolled back. The server listens on 127.0.0.1 only and rejects requests from other origins.

## The runner on its own

`runner.py` executes a plan without the app:

```
cd your-project
python path/to/runner.py plan.json [--retries 2] [--dry-run]
```

The plan format is documented at the top of `runner.py`. Check commands run in your shell
exactly as written, so only run plans you trust.

## Limits

- The local model handles small, precisely specified tasks. Planning and review need Claude.
- Stopping a run and killing aider uses `taskkill`, so the Stop button is Windows-only.
- One run at a time, and no resume: after a partial run, ask for a new plan.
