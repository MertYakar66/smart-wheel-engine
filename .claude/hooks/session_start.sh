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
try:
    from engine.wheel_runner import WheelRunner
    cls = type(WheelRunner().connector).__name__
    expected = "MarketDataConnector"  # bloomberg provider
    mark = "✓" if cls == expected else "⚠"
    print(f"{mark} connector class = {cls} (expected {expected})")
except Exception as e:
    print(f"✗ connector smoke failed: {type(e).__name__}: {e}")
PY

echo "└────────────────────────────────────────────────────────────────"
