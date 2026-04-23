@echo off
REM ============================================================
REM  smart-wheel-engine — ThetaData bulk backfill
REM  Double-click to pull every dataset the engine needs.
REM  The ThetaTerminal must be running first.
REM ============================================================

cd /d "%~dp0"

echo.
echo [1/3] Checking ThetaTerminal health...
python scripts\theta_health_check.py
if errorlevel 1 (
    echo.
    echo   Terminal not healthy. Start it with:
    echo     java -jar ThetaTerminalv3.jar ^<email^> ^<password^>
    echo   then run this script again.
    pause
    exit /b 1
)

echo.
echo [2/3] Running full backfill (this can take 30-90 minutes)...
echo   Output: data_processed\theta\
echo   Progress will print every 25 tickers.
python -m scripts.theta_backfill all
if errorlevel 1 (
    echo.
    echo   Backfill reported failures. Check the log above.
    echo   Re-run is safe — already-pulled files are skipped.
    pause
    exit /b 1
)

echo.
echo [3/3] Done. Manifest written to data_processed\theta\_manifest.json
echo.
pause
