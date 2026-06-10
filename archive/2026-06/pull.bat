@echo off
REM ============================================================
REM  smart-wheel-engine updater
REM  Double-click to sync local code with origin/main.
REM  Stash any local edits first so you never lose work.
REM ============================================================

cd /d "%~dp0"

echo.
echo [1/4] Stashing any local changes (if present)...
git stash push -u -m "auto-stash before pull %date% %time%" 2>nul

echo.
echo [2/4] Fetching and pulling from origin/main...
git checkout main
git pull origin main

echo.
echo [3/4] Restoring stashed changes (if any)...
git stash list | findstr "auto-stash" >nul && git stash pop
if errorlevel 1 echo    (nothing to restore)

echo.
echo [4/4] Done. Current commit:
git log --oneline -1

echo.
pause
