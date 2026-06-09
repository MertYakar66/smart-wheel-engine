"""Database-integrity tests for the bundled Bloomberg connector CSVs.

Phase 2 (A) of the data + engine audit (Phase 1 = ``scripts/audit_data_engine.py``
+ ``docs/DATA_ENGINE_AUDIT_2026-06-07.md``). These assert the DATA CONTRACT on
the **real committed files** in ``data/bloomberg/`` — distinct from the
synthetic-fixture coverage in ``tests/test_data_connector.py`` (which writes its
own tmp_path CSVs and never touches the bundled data). Skipped when the bundled
CSVs are absent, mirroring the ``HAS_BLOOMBERG_DATA`` skipif in
``tests/test_data_integration.py``.

Policy: a test that fails because the data is genuinely bad is a SUCCESS. We do
NOT soften assertions to go green. A confirmed data defect is ``xfail(strict=True)``
with a linked issue, so CI stays green, the defect is tracked, and the xfail
flips red the day it is fixed. Findings map to the audit's W-numbers.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector, normalize_ticker

DATA_DIR = Path("data/bloomberg")
HAS_BLOOMBERG_DATA = (DATA_DIR / "sp500_ohlcv.csv").exists()

pytestmark = pytest.mark.skipif(
    not HAS_BLOOMBERG_DATA, reason="Bloomberg CSVs not available (data/bloomberg)"
)

# Data-supported frontier (the most-recent bar common to OHLCV & IV on main;
# see the audit). Used to assert realized daily series carry no future bars.
FRONTIER = pd.Timestamp("2026-06-04")
SEAM = pd.Timestamp("2026-03-23")  # 2026-03-23 index-reconstitution seam
GATE_DAYS = 504  # ranker survivorship/history gate

# Files keyed by connector _FILES key.
_F = MarketDataConnector._FILES

# Canonical GICS 11 sectors — the only valid values of gics_sector_name and the
# grouping keys a GICS-aware R9 would use (see W17 / issue #372).
GICS_11 = {
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
}


@cache
def _load(fname: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / fname, low_memory=False)


@cache
def _keys(fname: str) -> pd.DataFrame:
    """date+ticker only (cheap on the big files)."""
    df = pd.read_csv(DATA_DIR / fname, usecols=lambda c: c in ("date", "ticker"))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@cache
def _norm_tickers(fname: str) -> frozenset[str]:
    df = pd.read_csv(DATA_DIR / fname, usecols=lambda c: c == "ticker")
    if "ticker" not in df.columns:
        return frozenset()
    return frozenset(df["ticker"].dropna().map(normalize_ticker).unique())


@cache
def _ohlcv_canonical() -> pd.DataFrame:
    """OHLCV with the connector's documented column rename applied
    (CSV open=HIGH, high=CLOSE, close=OPEN). Mirrors get_ohlcv."""
    df = _load("sp500_ohlcv.csv").rename(columns={"open": "high", "high": "close", "close": "open"})
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@cache
def _ohlcv_history_profile() -> pd.DataFrame:
    df = _keys("sp500_ohlcv.csv").copy()
    df["nt"] = pd.read_csv(DATA_DIR / "sp500_ohlcv.csv", usecols=["ticker"])["ticker"].map(
        normalize_ticker
    )
    g = df.groupby("nt")["date"]
    return pd.DataFrame({"first": g.min(), "last": g.max(), "ndays": g.count()})


# ---------------------------------------------------------------------------
# Schema / dtype
# ---------------------------------------------------------------------------

_EXPECTED_COLUMNS = {
    "sp500_ohlcv.csv": {"date", "ticker", "open", "high", "low", "close", "volume"},
    "sp500_vol_iv_full.csv": {
        "date",
        "ticker",
        "hist_put_imp_vol",
        "hist_call_imp_vol",
        "volatility_30d",
        "volatility_60d",
        "volatility_90d",
        "volatility_260d",
    },
    "sp500_dividends.csv": {"ticker", "ex_date", "dividend_amount"},
    "sp500_earnings.csv": {"ticker", "announcement_date"},
    "treasury_yields.csv": {"date", "rate_3m", "rate_10y"},
    "vix_term_structure.csv": {"date", "vix"},
    "sp500_fundamentals.csv": {"ticker", "gics_sector_name", "eqy_dvd_yld_12m"},
    "sp500_credit_risk.csv": {"ticker", "rtg_sp_lt_lc_issuer_credit"},
    "sp500_liquidity.csv": {"date", "ticker", "avg_vol_30d"},
}


@pytest.mark.parametrize("fname,required", sorted(_EXPECTED_COLUMNS.items()))
def test_schema_required_columns(fname, required):
    """Every connector CSV carries the columns its accessor depends on."""
    cols = set(_load(fname).columns)
    missing = required - cols
    assert not missing, f"{fname} missing required columns {missing}"


def test_all_connector_files_present():
    """Every file the connector will try to read exists on disk."""
    missing = [fn for fn in _F.values() if not (DATA_DIR / fn).exists()]
    assert not missing, f"connector _FILES missing on disk: {missing}"


# ---------------------------------------------------------------------------
# OHLCV integrity
# ---------------------------------------------------------------------------


def test_ohlcv_prices_positive():
    """All non-NaN OHLC prices are strictly positive."""
    df = _ohlcv_canonical()
    ocl = df[["open", "high", "low", "close"]]
    nonpos = (ocl.dropna() <= 0).any(axis=1).sum()
    assert nonpos == 0, f"{nonpos} OHLCV rows have a non-positive price"


def test_ohlcv_volume_nonnegative():
    df = _ohlcv_canonical()
    v = df["volume"].dropna()
    assert (v < 0).sum() == 0, "negative volume present"


def test_ohlcv_rename_invariant_holds():
    """The load-bearing column-rename invariant: high >= max(o,c,l) and
    low <= min(o,c,h) on EVERY row. If a future CSV ships in canonical order
    the rename would silently invert price history (connector warns; this
    fails loudly). Verified 0 violations / 1.01M rows in the audit."""
    df = _ohlcv_canonical().dropna(subset=["open", "high", "low", "close"])
    bad_high = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    assert bad_high == 0 and bad_low == 0, (
        f"rename invariant broken: high<max(o,c,l)={bad_high}, low>min(o,c,h)={bad_low} "
        "— the Bloomberg column-rotation assumption may have drifted"
    )


@pytest.mark.xfail(
    strict=True,
    reason="W10/W11 hygiene (#357): 4 vendor-glitch NaN-price rows "
    "(BIIB 2020-11-06/2023-06-09, TPL 2019-05-16/2019-07-09) — price NaN, volume present",
)
def test_ohlcv_no_nan_prices():
    """OHLC prices must be complete. Currently fails on 4 vendor-glitch rows."""
    df = _ohlcv_canonical()
    nan_rows = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
    assert nan_rows == 0, f"{nan_rows} OHLCV rows have NaN prices"


def test_ohlcv_keys_unique_and_no_future_bars():
    df = _keys("sp500_ohlcv.csv")
    dups = df.duplicated(subset=["date", "ticker"]).sum()
    assert dups == 0, f"{dups} duplicate (date,ticker) OHLCV rows"
    future = (df["date"] > FRONTIER + pd.Timedelta(days=3)).sum()
    assert future == 0, f"{future} OHLCV bars dated beyond the frontier (look-ahead)"


# ---------------------------------------------------------------------------
# vol_iv integrity (units, band, zero-skew) — ties W1 + W8 + W9
# ---------------------------------------------------------------------------


def test_vol_iv_unit_consistency_and_band():
    """IV is internally consistent in PERCENT units, finite, and within an
    absolute-plausible band. Surfaces the sane-band outliers (W8) and the
    sub-3.0 rows the ranker's `iv>3.0` heuristic would mis-scale (W1, #356)."""
    df = _load("sp500_vol_iv_full.csv")
    p = pd.to_numeric(df["hist_put_imp_vol"], errors="coerce")
    c = pd.to_numeric(df["hist_call_imp_vol"], errors="coerce")
    both = pd.concat([p, c]).dropna()

    # Unit consistency: a percent panel has a median in the tens. If it were
    # accidentally stored decimal the median would be ~0.28 -> this catches it.
    median = float(both.median())
    assert 5.0 < median < 100.0, f"IV median {median} inconsistent with PERCENT units"

    # Absolute-impossible bound (catches sentinels like 134217.7, negatives, 0).
    assert (both <= 0).sum() == 0, "non-positive IV present"
    assert (both > 1000).sum() == 0, f"IV > 1000% present (max={both.max()}) — possible sentinel"

    # Sane-band outliers surfaced and pinned small (W8). Currently 7.
    out_of_band = int(((both < 0.1) | (both > 500)).sum())
    assert out_of_band <= 20, (
        f"{out_of_band} IV rows outside the sane band [0.1%,500%] "
        "(implausible extremes) — investigate before it grows"
    )

    # W1 heuristic-exposure surface: rows the ranker's `if iv>3.0: iv/100`
    # would leave UNDIVIDED (read up to 300%). Pinned small; the fix is #356.
    danger = int(((both > 0) & (both <= 3.0)).sum())
    assert danger <= 50, (
        f"{danger} vol_iv rows have 0<IV<=3.0 — the ranker heuristic (#356) "
        "would mis-scale these; the count grew unexpectedly"
    )


def test_vol_iv_zero_put_call_skew():
    """W9: put_iv == call_iv EXACTLY across 100% of both-present rows — the
    Bloomberg surface is flat (skew dormant). Pinned so a future skew-bearing
    refresh is noticed (the Nelson-Siegel tooling silently consumes a flat IV)."""
    df = _load("sp500_vol_iv_full.csv")
    p = pd.to_numeric(df["hist_put_imp_vol"], errors="coerce")
    c = pd.to_numeric(df["hist_call_imp_vol"], errors="coerce")
    both = p.notna() & c.notna()
    eq_frac = float((p[both] == c[both]).mean())
    assert eq_frac == 1.0, (
        f"put_iv==call_iv only {eq_frac:.4%} (skew now present — re-check skew path)"
    )


def test_vol_iv_keys_unique_no_future():
    df = _keys("sp500_vol_iv_full.csv")
    assert df.duplicated(subset=["date", "ticker"]).sum() == 0
    assert (df["date"] > FRONTIER + pd.Timedelta(days=3)).sum() == 0


# ---------------------------------------------------------------------------
# Served vol_iv band gate (#363) + realized-vol sanity — W14 / W21 / W26
# (2026-06-09 data-test audit; docs/DATA_TEST_AUDIT_2026-06-09.md)
#
# The tests above read the RAW CSV via pd.read_csv. These read through the
# CONNECTOR — the IV the engine actually sees, AFTER #363's gate.
# ---------------------------------------------------------------------------

_IV_LEGS = ("hist_put_imp_vol", "hist_call_imp_vol")


def test_connector_iv_band_constants_are_authoritative():
    """W26: the #363 band is the AUTHORITATIVE Bloomberg-served IV rule. Pinning
    the constants means a silent floor/ceiling change fails CI. (A second,
    DIFFERENT IV-sanity heuristic lives in ``utils.data_validation`` but applies
    only to the Theta option-chain path — never the connector-served path here.)"""
    assert MarketDataConnector._IV_LOW_FLOOR == 3.0
    assert MarketDataConnector._DEEP_IV_SENTINEL_FLOOR == 10000.0


def test_served_vol_iv_band_via_connector():
    """W14: every IV the engine SEES (post-#363 ``_clean_vol_iv_inplace``) is in
    ``(3.0, 10000]``. Asserted on the connector's SERVED read of the bundled file
    — not the raw CSV (``test_vol_iv_unit_consistency_and_band`` covers the raw
    file, which still carries the sub-3.0 garbage the gate removes at read time).
    A regression in the gate (floor flipped, helper skipped on the monolith)
    fails here even though the raw file is unchanged."""
    served = MarketDataConnector()._load("vol_iv")
    for col in _IV_LEGS:
        s = pd.to_numeric(served[col], errors="coerce").dropna()
        assert len(s) > 0, f"{col}: no served IV present"
        assert (s <= 3.0).sum() == 0, f"{col}: served IV <= 3.0 present (low-floor gate breached)"
        assert (s > 10000.0).sum() == 0, f"{col}: served IV > 10000 present (sentinel leaked)"
        assert s.min() > 3.0


def test_served_vol_iv_gate_removes_raw_sub3():
    """W14: the GATE (not the file) is what removes the sub-3.0 garbage — the raw
    file still carries it, the served read does not. Robust to a future clean
    refresh: the delta is only asserted while the raw file still has sub-3.0 rows
    (the served==0 invariant holds either way)."""
    raw = _load("sp500_vol_iv_full.csv")
    served = MarketDataConnector()._load("vol_iv")
    for col in _IV_LEGS:
        raw_num = pd.to_numeric(raw[col], errors="coerce")
        served_num = pd.to_numeric(served[col], errors="coerce")
        raw_sub3 = int(((raw_num > 0) & (raw_num <= 3.0)).sum())
        assert (served_num <= 3.0).sum() == 0, f"{col}: served still has sub-3.0 IV"
        if raw_sub3 > 0:
            assert served_num.notna().sum() < raw_num.notna().sum(), (
                f"{col}: raw has {raw_sub3} sub-3.0 cells but the served read nulled none"
            )


def test_vol_iv_realized_vol_columns_positive_finite():
    """W21: the realized-vol columns (``volatility_30d/60d/90d/260d``, which feed
    the F4 RV-widening signal) are positive + finite where present on the real
    file. A corrupt/zero/negative realized vol would otherwise reach the
    forward-distribution tail with no failing test."""
    df = _load("sp500_vol_iv_full.csv")
    inf = float("inf")
    for col in ("volatility_30d", "volatility_60d", "volatility_90d", "volatility_260d"):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        assert len(s) > 0, f"{col}: no values present"
        assert ((s != inf) & (s != -inf)).all(), f"{col}: non-finite realized vol present"
        assert (s > 0).all(), f"{col}: non-positive realized vol present (min={s.min()})"


# ---------------------------------------------------------------------------
# Dividends (W11)
# ---------------------------------------------------------------------------


def test_dividends_nonnegative():
    """Dividend amounts are non-negative within float tolerance. The 82
    Discontinued/Omitted rows are -2.4e-14 epsilon noise (>= -1e-9); a true
    negative dividend would fail. Producer-clamp tracked in #357."""
    df = _load("sp500_dividends.csv")
    da = pd.to_numeric(df["dividend_amount"], errors="coerce")
    materially_negative = int((da < -1e-9).sum())
    assert materially_negative == 0, f"{materially_negative} materially-negative dividend amounts"


def test_dividends_tickers_resolve_in_ohlcv():
    """Every dividend payer exists in the OHLCV spine."""
    missing = sorted(_norm_tickers("sp500_dividends.csv") - _norm_tickers("sp500_ohlcv.csv"))
    assert not missing, f"dividend tickers absent from OHLCV: {missing}"


# ---------------------------------------------------------------------------
# Treasury (W10)
# ---------------------------------------------------------------------------


def test_treasury_band_and_coverage():
    """rate_3m within a plausible band that ALLOWS the brief negative short-bill
    episodes (2008/2011/2015/2020/2021, min ~-0.14%) but flags impossibles.
    Also pins that coverage starts in 1994 — memory's 'treasury only 2021-05+'
    is STALE (R1/#338 backfill)."""
    df = _load("treasury_yields.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    r3 = pd.to_numeric(df["rate_3m"], errors="coerce").dropna()
    assert r3.min() >= -0.5, f"rate_3m min {r3.min()} below plausible floor (-0.5%)"
    assert r3.max() <= 25.0, f"rate_3m max {r3.max()} above plausible ceiling (25%)"
    first = df.loc[pd.to_numeric(df["rate_3m"], errors="coerce").notna(), "date"].min()
    assert first <= pd.Timestamp("2018-01-01"), (
        f"treasury rate_3m starts {first} — expected pre-2018 coverage "
        "(memory's '2021-05+' is stale; R1/#338 backfilled to 1994)"
    )


# ---------------------------------------------------------------------------
# Date hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fname", ["sp500_ohlcv.csv", "sp500_vol_iv_full.csv", "sp500_liquidity.csv"]
)
def test_daily_series_dates_sorted_per_ticker(fname):
    """Within each ticker the daily series is strictly increasing in date
    (sorted + unique). Connector accessors sort, but a malformed file would
    break PIT `<= as_of` slicing semantics."""
    df = _keys(fname).copy()
    df["nt"] = pd.read_csv(DATA_DIR / fname, usecols=["ticker"])["ticker"].map(normalize_ticker)
    # group max-count of duplicate dates per ticker
    dup = df.duplicated(subset=["nt", "date"]).sum()
    assert dup == 0, f"{fname}: {dup} duplicate (ticker,date) rows"


# ---------------------------------------------------------------------------
# Cross-file referential integrity + the seam membership split (W4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fname", ["sp500_vol_iv_full.csv", "sp500_earnings.csv", "sp500_liquidity.csv"]
)
def test_ticker_files_subset_of_ohlcv(fname):
    """Every ticker-keyed file resolves into the OHLCV spine (no orphan names)."""
    missing = sorted(_norm_tickers(fname) - _norm_tickers("sp500_ohlcv.csv"))
    assert not missing, f"{fname} tickers absent from OHLCV spine: {missing}"


def test_fundamentals_credit_subset_of_ohlcv():
    spine = _norm_tickers("sp500_ohlcv.csv")
    assert not (_norm_tickers("sp500_fundamentals.csv") - spine)
    assert not (_norm_tickers("sp500_credit_risk.csv") - spine)


def test_seam_membership_split_is_structural():
    """W4: the OHLCV-vs-other gaps are NOT random — they are the documented
    2026-03-23 reconstitution. Encode the expectation structurally (not as a
    failure): names in OHLCV-but-not-earnings are post-seam JOINERS; names in
    OHLCV-but-not-fundamentals are pre-seam LEAVERS."""
    prof = _ohlcv_history_profile()
    spine = _norm_tickers("sp500_ohlcv.csv")

    joiners = sorted(spine - _norm_tickers("sp500_earnings.csv"))
    for t in joiners:
        first = prof.loc[t, "first"]
        assert first >= SEAM, (
            f"{t} is missing from earnings but is NOT a post-seam joiner "
            f"(first bar {first}) — unexpected referential gap"
        )

    leavers = sorted(spine - _norm_tickers("sp500_fundamentals.csv"))
    for t in leavers:
        last = prof.loc[t, "last"]
        assert last < FRONTIER - pd.Timedelta(days=30), (
            f"{t} is missing from fundamentals but is NOT a departed/leaver name "
            f"(last bar {last}) — unexpected referential gap"
        )


def test_seam_continuity_and_reticker():
    """2026-03-23 seam: per-day counts step cleanly (no dup); the BK->BNY
    re-ticker is present as BK ending 2026-03-20 + BNY starting 2026-03-23."""
    keys = _keys("sp500_ohlcv.csv").copy()
    keys["nt"] = pd.read_csv(DATA_DIR / "sp500_ohlcv.csv", usecols=["ticker"])["ticker"].map(
        normalize_ticker
    )
    around = keys[(keys["date"] >= "2026-03-18") & (keys["date"] <= "2026-03-26")]
    per_day = around.groupby("date")["nt"].nunique()
    # No date in the seam window has zero or duplicated names.
    assert (per_day > 0).all()
    assert around.duplicated(subset=["date", "nt"]).sum() == 0
    prof = _ohlcv_history_profile()
    assert "BK" in prof.index and "BNY" in prof.index, "BK->BNY re-ticker names missing"
    assert prof.loc["BK", "last"] <= SEAM, "BK should end at/before the seam"
    assert prof.loc["BNY", "first"] >= SEAM, "BNY should start at/after the seam"


# ---------------------------------------------------------------------------
# Fingerprint completeness (durable W3 replacement) — fast-CI guard
# ---------------------------------------------------------------------------


def test_fingerprint_pins_every_connector_file():
    """The snapshot fingerprint must pin EXACTLY the files the connector reads,
    so no un-pinned read path can silently slip a refresh past re-baseline (the
    2026-06-06 dividends-incident class). This is the FAST-CI completeness guard;
    the drift COMPARE (test_snapshot_data_fingerprint_matches_current) lives on
    the slow backtest_regression lane."""
    from backtests.regression._common import connector_data_sha256

    pinned = set(connector_data_sha256().keys())
    expected = set(MarketDataConnector._FILES.keys())
    assert pinned == expected, (
        f"fingerprint pins {pinned} but connector reads {expected}; unpinned: {expected - pinned}"
    )


# ---------------------------------------------------------------------------
# Fundamentals content + R9 sector-map coverage — W19 / W17
# (2026-06-09 data-test audit; docs/DATA_TEST_AUDIT_2026-06-09.md)
# ---------------------------------------------------------------------------


def test_fundamentals_dividend_yield_in_band():
    """W19: eqy_dvd_yld_12m (the BSM carry-q source, PERCENT units) is non-negative
    and within a plausible band on the real file. A negative or absurd yield would
    corrupt the BSM carry. NaN is allowed — ~95 names lack a trailing 12m yield."""
    df = _load("sp500_fundamentals.csv")
    y = pd.to_numeric(df["eqy_dvd_yld_12m"], errors="coerce").dropna()
    assert len(y) > 0, "no dividend-yield values present"
    assert (y >= 0).all(), f"negative dividend yield present (min={y.min()})"
    assert (y <= 30).all(), f"implausible dividend yield > 30% present (max={y.max()})"


def test_fundamentals_gics_sector_is_canonical_11():
    """W19: gics_sector_name is a subset of the canonical GICS 11. A 12th/typo'd
    sector would silently mis-bucket screen_universe (and any future GICS-grouped
    R9 — see W17 / #372)."""
    df = _load("sp500_fundamentals.csv")
    sec = {s for s in df["gics_sector_name"].dropna().astype(str).unique() if s != "nan"}
    extra = sec - GICS_11
    assert not extra, f"non-canonical GICS sector(s) present: {sorted(extra)}"


def test_r9_sector_map_ignores_pulled_gics_characterization():
    """W17 (#372): R9's sector cap groups by the HARDCODED ``DEFAULT_SECTOR_MAP``,
    NOT the pulled ``gics_sector_name``. Characterise the gap on real data: many
    names carry a real GICS sector in fundamentals yet
    ``SectorExposureManager.get_sector`` returns ``'Unknown'`` (so R9 lumps them
    into one phantom bucket). PASSING today — it flips when #372 wires GICS into
    R9, at which point update it to assert the GICS-grouped behaviour. Quantifies
    the coverage gap so a map drift is noticed."""
    from engine.risk_manager import SectorExposureManager

    fu = _load("sp500_fundamentals.csv")
    fu = fu.assign(nt=fu["ticker"].map(normalize_ticker))
    has_gics = fu[fu["gics_sector_name"].notna() & fu["gics_sector_name"].astype(str).ne("nan")]
    mgr = SectorExposureManager()
    ignored = [t for t in has_gics["nt"] if mgr.get_sector(t) == "Unknown"]
    assert len(ignored) > 50, (
        f"expected many GICS-known names bucketed as 'Unknown' by R9 (the #372 gap); "
        f"got {len(ignored)} — if near 0, R9 may now read GICS: update this "
        "characterization to assert the GICS-grouped behaviour"
    )


# ---------------------------------------------------------------------------
# OHLCV depth + NaN-price pins + treasury rate_1m residual — W22 / W23 / W10
# (2026-06-09 data-test audit; docs/DATA_TEST_AUDIT_2026-06-09.md)
# ---------------------------------------------------------------------------

# 17 names below the 504-bar survivorship gate at the 2026-06-04 frontier:
# W6 backfill defects (#355) + legit-recent joiners. A NEW name dropping below the
# gate (outside this set) is the silent truncation W22 catches.
KNOWN_THIN = {
    "WMT",
    "KMB",
    "CPB",
    "DPZ",
    "PLTR",
    "VEEV",
    "COHR",
    "CASY",
    "LITE",
    "SATS",
    "VRT",
    "BNY",
    "FDXF",
    "SNDK",
    "SW",
    "PSKY",
    "Q",
}

# The 4 vendor-glitch NaN-price rows (price NaN, volume present) — #357.
_KNOWN_NAN_PRICE_ROWS = {
    ("BIIB", "2020-11-06"),
    ("BIIB", "2023-06-09"),
    ("TPL", "2019-05-16"),
    ("TPL", "2019-07-09"),
}


def test_ohlcv_per_name_depth_invariant():
    """W22: every name with < 504 OHLCV bars (the survivorship gate) is a KNOWN
    thin name. A NEW blue-chip silently truncated < 504 — outside the known set —
    fails here, instead of being invisible (the only other depth check is the
    per-name W6 xfail in test_data_to_engine.py, which skips off-frontier and only
    covers 11 named tickers)."""
    prof = _ohlcv_history_profile()
    thin = {t for t in prof.index if int(prof.loc[t, "ndays"]) < GATE_DAYS}
    unexpected = thin - KNOWN_THIN
    assert not unexpected, (
        f"NEW <504-bar OHLCV names outside the known thin set: {sorted(unexpected)} "
        "(a silent blue-chip truncation — investigate before it reaches the gate)"
    )


def test_ohlcv_nan_price_rows_are_the_known_four():
    """W23: pin the 4 vendor-glitch NaN-price rows two-sidedly. The xfail
    test_ohlcv_no_nan_prices flips when they are FIXED (count -> 0); THIS catches
    the count GROWING — a 5th NaN row from a future refresh fails here (the
    one-directional xfail would stay green). Pins the exact (ticker,date) keys."""
    df = _ohlcv_canonical()
    nan_mask = df[["open", "high", "low", "close"]].isna().any(axis=1)
    sub = df.loc[nan_mask]
    found = {
        (normalize_ticker(str(t)), str(d)[:10])
        for t, d in zip(sub["ticker"], sub["date"], strict=False)
    }
    assert found == _KNOWN_NAN_PRICE_ROWS, (
        f"OHLCV NaN-price rows changed: {sorted(found)} != known "
        f"{sorted(_KNOWN_NAN_PRICE_ROWS)} — a 5th vendor-glitch NaN would slip past "
        "the one-directional xfail; investigate + update the pin"
    )


def test_treasury_rate_1m_coverage_gap_and_nan_before():
    """W10 residual: rate_1m has a documented coverage GAP — it starts LATER than
    rate_3m (the audit found 2001-07 vs 1994) — and, crucially, an as_of BEFORE its
    coverage returns NaN (never a spurious 0% rate); values sit in a plausible band
    that ALLOWS the brief negative-bill episodes. (test_treasury_band_and_coverage
    pins rate_3m; this closes the rate_1m gap the audit flagged.) The 'before' as_of
    is derived from the actual coverage start, so a future backfill can't false-fail
    it — the safety property (NaN, not 0) is what's pinned, not a hard date."""
    df = _load("treasury_yields.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    r1 = pd.to_numeric(df["rate_1m"], errors="coerce")
    r3 = pd.to_numeric(df["rate_3m"], errors="coerce")
    first_1m = df.loc[r1.notna(), "date"].min()
    first_3m = df.loc[r3.notna(), "date"].min()
    assert first_1m > first_3m, (
        f"rate_1m first non-null {first_1m} not later than rate_3m {first_3m} "
        "(the documented W10 coverage gap closed — update the pin)"
    )
    nn = r1.dropna()
    assert nn.min() >= -0.5, f"rate_1m min {nn.min()} below plausible floor (-0.5%)"
    assert nn.max() <= 25.0, f"rate_1m max {nn.max()} above plausible ceiling (25%)"
    # the safety property: an as_of BEFORE rate_1m coverage -> NaN, never a spurious 0
    before = str((first_1m - pd.Timedelta(days=30)).date())
    early = MarketDataConnector().get_risk_free_rate(before, tenor="rate_1m")
    assert pd.isna(early), f"rate_1m before coverage ({before}) should be NaN, got {early}"


# ---------------------------------------------------------------------------
# Credit content (OFF the EV path — display/legacy heuristic only, capability C1) — W24
# (2026-06-09 data-test audit; docs/DATA_TEST_AUDIT_2026-06-09.md)
# ---------------------------------------------------------------------------

# S&P long-term issuer credit ladder. The raw field carries CreditWatch/outlook
# suffixes (' *-' negative watch, ' *+' positive watch) that a naive parse rejects.
_SP_LADDER = {
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
    "CC",
    "C",
    "RD",
    "SD",
    "D",
}
_SP_NONRATED = {"NR", "N.A.", "NA", "NONE", ""}


def test_credit_sp_rating_is_valid_ladder():
    """W24: every sp_rating, after stripping the CreditWatch suffix (' *-'/' *+'),
    is a member of the S&P long-term ladder (AAA..D) or a non-rated sentinel (NR).
    The raw field carries suffixed values ('A *-', 'CCC+ *+') a naive parse would
    reject — pins they normalise cleanly and that a future malformed rating is
    noticed. NOTE: credit is OFF the EV-authoritative path (feeds the legacy
    heuristic + display only, capability C1) — display-severity, not an EV gate."""
    df = _load("sp500_credit_risk.csv")
    rt = df["rtg_sp_lt_lc_issuer_credit"].dropna().astype(str).str.strip()
    base = rt.str.replace(r"\s*\*[-+]$", "", regex=True).str.strip().str.upper()
    ladder = {s.upper() for s in _SP_LADDER}
    nonrated = {s.upper() for s in _SP_NONRATED} | {"NAN"}
    bad = sorted({b for b in base.unique() if b not in ladder and b not in nonrated})
    assert not bad, f"sp_rating values outside the S&P ladder after suffix-strip: {bad}"


def test_credit_altman_z_plausible_band():
    """W24: altman_z_score sits in a wide plausibility band and its negative count is
    bounded. (2 values >100 are known off-EV-path artifacts — financials/REITs where
    Altman-Z is not meaningful; the band accepts them while catching a producer
    regression that floods negatives or absurd magnitudes.) Off-EV path → LOW."""
    df = _load("sp500_credit_risk.csv")
    z = pd.to_numeric(df["altman_z_score"], errors="coerce").dropna()
    assert len(z) > 0, "no altman_z values present"
    assert z.min() >= -10.0, f"altman_z min {z.min()} below plausible floor (-10)"
    assert z.max() <= 200.0, f"altman_z max {z.max()} above plausible ceiling (200)"
    assert int((z < 0).sum()) <= 10, (
        f"too many negative altman_z ({int((z < 0).sum())}) — possible producer regression"
    )
