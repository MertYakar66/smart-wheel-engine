#!/usr/bin/env bash
# Smart Wheel Engine — SessionStart hook for Cowork / fresh Claude sessions.
# Idempotent: re-running is fast when everything is already in place.
# Never exits non-zero — individual check failures print warnings and
# continue so the session can still start.

set +e

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO" || exit 0

echo "┌─ Smart Wheel Engine — SessionStart ─────────────────────────────"

# 1. Provider selection — must be explicit, default to bloomberg.
if [ -z "$SWE_DATA_PROVIDER" ]; then
  export SWE_DATA_PROVIDER=bloomberg
  echo "│  ⚠ SWE_DATA_PROVIDER was unset — defaulted to 'bloomberg'"
  echo "│    (set it explicitly in your shell to silence this warning)"
else
  echo "│  ✓ SWE_DATA_PROVIDER=$SWE_DATA_PROVIDER (explicit)"
fi

# 2. Bloomberg CSVs — required for the bloomberg provider path.
for f in data/bloomberg/sp500_ohlcv.csv data/bloomberg/sp500_vol_iv_full.csv; do
  if [ -f "$f" ]; then
    echo "│  ✓ $(basename "$f") ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "│  ✗ missing: $f — bloomberg path will error"
  fi
done

# 2b. OHLCV staleness — warn-only when the most recent date in
#     sp500_ohlcv.csv is more than 30 days behind today. The Bloomberg
#     refresh path is documented as "monthly manual" (docs/DATA_POLICY.md
#     §5); without this check a stale CSV silently produces analyses that
#     look "current" but reason about old market state. Pure bash + awk +
#     GNU date so the warning fires the same on Cowork (Linux) and on a
#     Windows Git Bash dev box (where `python3` is the MS Store stub).
OHLCV="data/bloomberg/sp500_ohlcv.csv"
if [ -f "$OHLCV" ]; then
  # ISO YYYY-MM-DD in column 1 sorts lexically -> string-max == date-max.
  LAST_DATE=$(awk -F, 'NR>1 && $1>m {m=$1} END {print m}' "$OHLCV" 2>/dev/null)
  if [ -n "$LAST_DATE" ]; then
    LAST_EPOCH=$(date -d "$LAST_DATE" +%s 2>/dev/null)
    NOW_EPOCH=$(date +%s 2>/dev/null)
    if [ -n "$LAST_EPOCH" ] && [ -n "$NOW_EPOCH" ]; then
      DAYS_STALE=$(( (NOW_EPOCH - LAST_EPOCH) / 86400 ))
      if [ "$DAYS_STALE" -gt 30 ]; then
        echo "│  ⚠ sp500_ohlcv.csv is $DAYS_STALE days stale (most recent: $LAST_DATE)"
        echo "│    Refresh per docs/DATA_POLICY.md §5 before relying on a 'today' scan."
      fi
    fi
  fi
fi

# 3. Theta manifest — warn-only. Cowork still functions on bloomberg fallback.
MANIFEST="data_processed/theta/_manifest.json"
if [ -f "$MANIFEST" ]; then
  LAST=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['runs'][-1]['ran_at'])" 2>/dev/null || echo "?")
  echo "│  ✓ theta manifest present — last run $LAST"
else
  echo "│  ○ theta manifest absent — bloomberg-only mode (normal in Cowork)"
fi

# 4. Python deps — install missing packages in batches that fit a 45s budget.
python3 - <<'PY' 2>&1 | sed 's/^/│  /'
import importlib.util, os, subprocess, sys

batches = [
    ["scipy"],
    ["statsmodels", "arch", "scikit-learn"],
    ["yfinance", "pydantic", "pyarrow", "pytest"],
]
alias = {"scikit-learn": "sklearn"}

# Compute the full missing set up front so we can early-exit before
# attempting any install. Avoids noise when all deps are already
# present (e.g. a developer with system pandas/scipy from another
# project).
all_pkgs = [p for batch in batches for p in batch]
missing_all = [p for p in all_pkgs if importlib.util.find_spec(alias.get(p, p)) is None]

if missing_all:
    in_venv = sys.prefix != sys.base_prefix
    allow_global = os.environ.get("SWE_ALLOW_GLOBAL_PIP") == "1"
    if not in_venv and not allow_global:
        print(f"⚠ Missing deps {missing_all} — no venv active, SWE_ALLOW_GLOBAL_PIP unset.")
        print("  Activate your venv first, OR if this is an ephemeral sandbox:")
        print("    export SWE_ALLOW_GLOBAL_PIP=1 && bash .claude/hooks/session_start.sh")
        sys.exit(0)

# Safe to install: either in a venv (--break-system-packages is a no-op
# inside one) or explicit opt-in for global install.
for batch in batches:
    missing = [p for p in batch if importlib.util.find_spec(alias.get(p, p)) is None]
    if not missing:
        print(f"✓ deps present: {' '.join(batch)}")
        continue
    print(f"⟳ installing: {' '.join(missing)}")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--break-system-packages",
         "--quiet", *missing],
        capture_output=True, text=True,
    )
    status = "✓" if r.returncode == 0 else "✗"
    print(f"{status} installed: {' '.join(missing)}")
    if r.returncode != 0 and r.stderr:
        print(f"    {r.stderr.splitlines()[-1][:120]}")
PY

# 5. Connector smoke — confirms wheel_runner.py:130 wiring picks the
#    expected provider. Don't hang the session if this fails; just log.
python3 - <<'PY' 2>&1 | sed 's/^/│  /'
import os
provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg").lower()
expected = "ThetaConnector" if provider == "theta" else "MarketDataConnector"
try:
    from engine.wheel_runner import WheelRunner
    cls = type(WheelRunner().connector).__name__
    mark = "✓" if cls == expected else "⚠"
    print(f"{mark} connector class = {cls} (expected {expected})")
except Exception as e:
    print(f"✗ connector smoke failed: {type(e).__name__}: {e}")
PY

# 6. Parallel-session coordination — surfaced every session so no terminal
#    branches without seeing the contract + who's already working.
echo "│  ─ Parallel sessions (docs/PARALLEL_SESSIONS.md) ─"
echo "│    • Work the task card the Major Session allocated — don't self-select."
echo "│    • Edit only your card's 'owns' files; decision-layer trio is CI-gated."
echo "│    • Sn / D-numbers are assigned at MERGE, not work-start."
echo "│    • Document your task in docs/worklog/ ('python scripts/new_worklog.py')."
# 6b. Per-terminal env — parallel pytest / engine_api runs need isolation
#     (separate port + coverage file + pytest cache). Warn if unset.
if [ -z "${SWE_API_PORT:-}" ] || [ -z "${COVERAGE_FILE:-}" ] || [ -z "${PYTEST_CACHE_DIR:-}" ]; then
  echo "│    ⚠ per-terminal env unset — 'source scripts/setup-terminal.sh <letter>'"
  echo "│      (isolates SWE_API_PORT / COVERAGE_FILE / PYTEST_CACHE_DIR)"
else
  echo "│    ✓ per-terminal env: port=$SWE_API_PORT cov=$COVERAGE_FILE"
fi
# Live board claims — best-effort, only when gh is present + authed. Never
# blocks the session: gh failures are swallowed, no network wait is forced.
if command -v gh >/dev/null 2>&1; then
  CLAIMS=$(gh issue view 113 --repo MertYakar66/smart-wheel-engine --json body --jq '.body' 2>/dev/null \
           | sed -n '/Live state/,/^## /p' | grep '^|' | head -6)
  if [ -n "$CLAIMS" ]; then
    echo "│    Live board (#113):"
    echo "$CLAIMS" | sed 's/^/│      /'
  fi
fi

# 7. Doc currency — warn when the temporal docs drift. Bash-native (awk/date/
#    grep, NO python3 — it is the MS-Store stub in Windows Git Bash). The full
#    check (CHANGELOG too) is scripts/check_doc_currency.py, run in CI.
PS="PROJECT_STATE.md"
if [ -f "$PS" ]; then
  PS_DATE=$(grep -m1 -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' "$PS" 2>/dev/null)
  if [ -n "$PS_DATE" ]; then
    PS_EPOCH=$(date -d "$PS_DATE" +%s 2>/dev/null)
    NOW_EPOCH=$(date +%s 2>/dev/null)
    if [ -n "$PS_EPOCH" ] && [ -n "$NOW_EPOCH" ]; then
      PS_AGE=$(( (NOW_EPOCH - PS_EPOCH) / 86400 ))
      if [ "$PS_AGE" -gt 21 ]; then
        echo "│  ⚠ PROJECT_STATE.md last updated $PS_DATE (${PS_AGE}d ago) — refresh its durable state."
      fi
    fi
  fi
fi

echo "└────────────────────────────────────────────────────────────────"
