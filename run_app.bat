@echo off
title Accessible Live Captions
echo Starting Accessible Live Captions...
"C:\salo\jeff\env_caption\scripts\python.exe" caption_app.py
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error code: %errorlevel%
    pause
)
