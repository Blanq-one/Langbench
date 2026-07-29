@echo off
rem Windows Task Scheduler entry point for the unattended daily pass (Part 3).
rem Registered as scheduled task "Langbench Daily Pass". `uv` is not on PATH
rem on this machine; enter the project venv via `python -m uv run`.
cd /d "%~dp0.."
python -m uv run python scripts\daily_pass.py
