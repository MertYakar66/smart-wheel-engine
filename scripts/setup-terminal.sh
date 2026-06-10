# shellcheck shell=bash
# scripts/setup-terminal.sh — per-terminal env for parallel Claude Code sessions.
#
# Source this file (don't execute it) so the exports persist in your shell:
#
#     source scripts/setup-terminal.sh a      # Terminal A → port 8787
#     source scripts/setup-terminal.sh b      # Terminal B → port 8788
#     source scripts/setup-terminal.sh c      # Terminal C → port 8789, etc.
#
# Companion: scripts/setup-terminal.ps1 (native PowerShell).
# Doc:        docs/PARALLEL_SESSIONS.md "Env vars per terminal".
#
# What this sets:
#
#   SWE_API_PORT                — 8787 + (letter - 'a'). Honoured today
#                                 by engine_api.py's `_resolve_port()` and
#                                 scripts/audit_api_smoke.py's `BASE` (PR #158 / D15);
#                                 default 8787 falls through when unset.
#   SWE_DATA_PROCESSED_DIR      — shared `data_processed/` by default.
#   SWE_MODELS_DIR              — shared `models/` by default.
#                                 Switch both to per-terminal (`data_processed_<x>/`
#                                 / `models_<x>/`) only if you hit write
#                                 contention; the engine does not read these
#                                 env vars yet, so these two remain a
#                                 convention until a consumer is wired in.
#   COVERAGE_FILE               — per-terminal `.coverage.<letter>`.
#                                 Read by coverage.py automatically — this
#                                 one is real today and stops parallel
#                                 pytest runs from stomping each other.
#   PYTEST_CACHE_DIR            — per-terminal `.pytest_cache_<letter>`.
#                                 Read by pytest automatically.
#   SWE_DATA_PROVIDER           — bloomberg (silences the SessionStart
#                                 warning; matches the project default
#                                 per CLAUDE.md §1).

if [ -z "${BASH_SOURCE:-}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: source this file, don't execute it:" >&2
    echo "  source scripts/setup-terminal.sh <letter>" >&2
    return 1 2>/dev/null || exit 1
fi

letter="${1:-}"
if [ -z "$letter" ]; then
    echo "ERROR: pass the terminal letter:" >&2
    echo "  source scripts/setup-terminal.sh a" >&2
    return 1
fi

letter="$(printf '%s' "$letter" | tr '[:upper:]' '[:lower:]')"
if ! printf '%s' "$letter" | grep -qE '^[a-z]$'; then
    echo "ERROR: terminal letter must be a single a-z character (got '$letter')." >&2
    return 1
fi

# Port = 8787 + (letter - 'a').
letter_ord="$(printf '%d' "'$letter")"
a_ord="$(printf '%d' "'a")"
port=$(( 8787 + letter_ord - a_ord ))

export SWE_API_PORT="$port"
export SWE_DATA_PROCESSED_DIR="data_processed/"
export SWE_MODELS_DIR="models/"
export COVERAGE_FILE=".coverage.$letter"
export PYTEST_CACHE_DIR=".pytest_cache_$letter"
export SWE_DATA_PROVIDER="bloomberg"

cat <<MSG
Terminal $letter env loaded:
  SWE_API_PORT           = $SWE_API_PORT
  SWE_DATA_PROCESSED_DIR = $SWE_DATA_PROCESSED_DIR
  SWE_MODELS_DIR         = $SWE_MODELS_DIR
  COVERAGE_FILE          = $COVERAGE_FILE
  PYTEST_CACHE_DIR       = $PYTEST_CACHE_DIR
  SWE_DATA_PROVIDER      = $SWE_DATA_PROVIDER
MSG

unset letter letter_ord a_ord port
