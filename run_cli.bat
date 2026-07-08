@echo off
title Accessible Live Captions CLI
echo Starting Accessible Live Captions CLI...
"C:\salo\jeff\env_caption\scripts\python.exe" caption_cli.py
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error code: %errorlevel%
    pause
)
