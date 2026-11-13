@echo off
rem Double-click to start the Hybrid app. Close this window to stop it.
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo This app needs a free tool called uv. Install it from https://docs.astral.sh/uv/ and try again.
  pause
  exit /b 1
)
rem open the browser after the server has had a few seconds to start
start "" /b cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8765"
uv run --python 3.12 hybrid_app.py
pause
