@echo off
REM Start the Thai News Intelligence Dashboard (backend + frontend).
REM
REM This wrapper exists because Windows blocks .ps1 files by default, so
REM `.\run.ps1` fails with "running scripts is disabled on this system".
REM -ExecutionPolicy Bypass applies to THIS invocation only: it changes no
REM machine or user setting, so nothing has to be loosened permanently.
REM
REM Double-click this file, or run it from a terminal. Arguments pass through:
REM     run.cmd -Reset
REM     run.cmd -NoBrowser

powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
