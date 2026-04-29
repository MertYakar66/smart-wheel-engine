@echo off
REM ============================================================
REM  smart-wheel-engine — pull the feature branch for testing
REM
REM  Double-click this INSTEAD of pull.bat when you want to test
REM  a feature branch before merging to main. It:
REM    1. Stashes any local edits (nothing gets lost)
REM    2. Fetches the feature branch from origin
REM    3. Checks it out locally (creates if it doesn't exist)
REM    4. Pulls the latest commits
REM    5. Restores your local edits
REM    6. Prints the current HEAD commit
REM
REM  Edit BRANCH_NAME below to point at whichever feature branch
REM  you want to test. Current default is the Claude dev branch.
REM ============================================================

set BRANCH_NAME=claude/map-codebase-architecture-aBvbq

cd /d "%~dp0"

echo.
echo [1/5] Stashing any local changes (if present)...
git stash push -u -m "auto-stash before branch pull %date% %time%" 2>nul

echo.
echo [2/5] Fetching %BRANCH_NAME% from origin...
git fetch origin %BRANCH_NAME%
if errorlevel 1 (
    echo   Fetch failed. Check your network and that the branch exists.
    pause
    exit /b 1
)

echo.
echo [3/5] Checking out %BRANCH_NAME%...
REM Create-and-track if the local branch doesn't exist; otherwise just switch.
git show-ref --verify --quiet refs/heads/%BRANCH_NAME%
if errorlevel 1 (
    echo   ^(creating local tracking branch^)
    git checkout -b %BRANCH_NAME% origin/%BRANCH_NAME%
) else (
    git checkout %BRANCH_NAME%
    git pull origin %BRANCH_NAME%
)
if errorlevel 1 (
    echo   Checkout failed.
    pause
    exit /b 1
)

echo.
echo [4/5] Restoring stashed changes (if any)...
git stash list | findstr "auto-stash before branch pull" >nul && git stash pop
if errorlevel 1 echo    ^(nothing to restore^)

echo.
echo [5/5] Done. You are now on branch:
git branch --show-current
echo.
echo Current HEAD commit:
git log --oneline -1
echo.
echo To switch back to main, run:
echo     git checkout main
echo     pull.bat
echo.
pause
