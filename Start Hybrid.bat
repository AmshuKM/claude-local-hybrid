@echo off
rem Windows: double-click to start the Hybrid app. Close this window to stop it.
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo This app needs a free tool called uv. Install it from https://docs.astral.sh/uv/ and try again.
  pause
  exit /b 1
)
uv run --python 3.12 hybrid_app.py --open
pause
