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
import importlib.util, subprocess, sys

batches = [
    ["scipy"],
    ["statsmodels", "arch", "scikit-learn"],
    ["yfinance", "pydantic", "pyarrow", "pytest"],
]
alias = {"scikit-learn": "sklearn"}

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
