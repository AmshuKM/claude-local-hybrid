#!/bin/sh
# macOS: double-click to start the Hybrid app (Linux: run ./"Start Hybrid.command").
# Close this window to stop it.
cd "$(dirname "$0")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
  echo "This app needs a free tool called uv. Install it from https://docs.astral.sh/uv/ and try again."
  read -r _
  exit 1
fi
exec uv run --python 3.12 hybrid_app.py --open
