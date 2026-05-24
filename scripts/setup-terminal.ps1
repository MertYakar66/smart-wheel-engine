# scripts/setup-terminal.ps1 — per-terminal env for parallel Claude Code sessions.
#
# Dot-source this file (don't execute it) so the env vars persist in your shell:
#
#     . .\scripts\setup-terminal.ps1 a        # Terminal A → port 8787
#     . .\scripts\setup-terminal.ps1 b        # Terminal B → port 8788
#     . .\scripts\setup-terminal.ps1 c        # Terminal C → port 8789, etc.
#
# Companion: scripts/setup-terminal.sh (bash / Git Bash / WSL).
# Doc:        docs/PARALLEL_SESSIONS.md "Env vars per terminal".
#
# Sets the same six vars as the bash companion; see that file's header for
# what each one means and which are real-today (SWE_API_PORT,
# COVERAGE_FILE, PYTEST_CACHE_DIR, SWE_DATA_PROVIDER) vs forward-looking
# convention (SWE_DATA_PROCESSED_DIR, SWE_MODELS_DIR).

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Letter
)

$Letter = $Letter.ToLower()
if ($Letter -notmatch '^[a-z]$') {
    Write-Error "Terminal letter must be a single a-z character (got '$Letter')."
    return
}

# Port = 8787 + (letter - 'a').
$port = 8787 + ([int][char]$Letter - [int][char]'a')

$env:SWE_API_PORT           = "$port"
$env:SWE_DATA_PROCESSED_DIR = "data_processed/"
$env:SWE_MODELS_DIR         = "models/"
$env:COVERAGE_FILE          = ".coverage.$Letter"
$env:PYTEST_CACHE_DIR       = ".pytest_cache_$Letter"
$env:SWE_DATA_PROVIDER      = "bloomberg"

Write-Host "Terminal $Letter env loaded:"
Write-Host "  SWE_API_PORT           = $env:SWE_API_PORT"
Write-Host "  SWE_DATA_PROCESSED_DIR = $env:SWE_DATA_PROCESSED_DIR"
Write-Host "  SWE_MODELS_DIR         = $env:SWE_MODELS_DIR"
Write-Host "  COVERAGE_FILE          = $env:COVERAGE_FILE"
Write-Host "  PYTEST_CACHE_DIR       = $env:PYTEST_CACHE_DIR"
Write-Host "  SWE_DATA_PROVIDER      = $env:SWE_DATA_PROVIDER"
