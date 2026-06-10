@echo off
REM ============================================================
REM  smart-wheel-engine — ThetaData bulk backfill
REM  Double-click to pull every dataset the engine needs.
REM  The ThetaTerminal must be running first.
REM ============================================================

cd /d "%~dp0"

echo.
echo [1/3] Probing ThetaTerminal endpoints...
python scripts\theta_health_check.py
set HEALTH_CODE=%errorlevel%

if %HEALTH_CODE% neq 0 (
    echo.
    echo   Some checks failed. This is usually a subscription-tier gap
    echo   (e.g. VIX-family endpoints need the Indices subscription),
    echo   not a Terminal problem. The engine falls back to CBOE public
    echo   data and Bloomberg CSVs for anything missing.
    echo.
    set /p CONTINUE="Continue with the backfill anyway? (y/N): "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

echo.
echo [2/3] Running full backfill (can take 30-90 minutes)...
echo   Output: data_processed\theta\
echo   Progress prints every 25 tickers. Safe to stop and restart.
python -m scripts.theta_backfill all
if errorlevel 1 (
    echo.
    echo   Backfill reported failures. Check the log above.
    echo   Re-run is safe — already-pulled files are skipped.
    pause
    exit /b 1
)

echo.
echo [3/3] Done. Manifest: data_processed\theta\_manifest.json
echo.
pause
