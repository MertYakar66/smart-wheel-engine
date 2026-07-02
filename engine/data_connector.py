"""
Unified Data Connector for Bloomberg Market Data

Loads, normalizes, and serves Bloomberg CSV data to all engine modules.
Handles ticker normalization (Bloomberg format -> standard),
date parsing, lazy loading with caching, and provides query methods
for OHLCV, volatility/IV, events, rates, fundamentals, and screening.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column contract for the option-premium rail (the produced per-ticker
# parquet AND the connector accessors below). Single source of truth — the
# producer (``scripts/produce_option_premiums.py``) imports this so the two
# never drift. ``mid = (bid + ask) / 2`` is the *real* EOD option premium that
# the (separate, EV-moving) ranker wiring will feed into
# ``ShortOptionTrade.premium`` so ``edge_vs_fair`` becomes non-zero — today the
# premium is synthetic-BSM and skew/VRP are EV-inert
# (docs/PHASE2_SKEW_EXECUTION_SPEC.md).
OPTION_PREMIUM_COLUMNS: tuple[str, ...] = (
    "date",  # EOD snapshot date (point-in-time axis)
    "expiration",
    "dte",
    "strike",
    "right",  # normalized to "put" / "call"
    "bid",
    "ask",
    "mid",
    "close",
    "volume",
    "open_interest",
)

# Bloomberg exchange suffixes to strip (exchange codes for US equities)
_EXCHANGE_SUFFIXES = {
    "UW",  # NASDAQ Global Select
    "UN",  # NYSE
    "UQ",  # NASDAQ Global Market
    "UA",  # NYSE American (AMEX)
    "UP",  # NYSE Arca
    "UR",  # NASDAQ Capital Market
    "US",  # generic US equity
}


def normalize_ticker(bbg_ticker: str) -> str:
    """Convert Bloomberg ticker to standard symbol.

    Examples:
        'AAPL UW Equity' -> 'AAPL'
        'A UN'           -> 'A'
        'BRK/B UN Equity'-> 'BRK/B'
        'AAPL'           -> 'AAPL'  (already standard)
    """
    if not isinstance(bbg_ticker, str):
        return str(bbg_ticker)

    ticker = bbg_ticker.strip()

    # Strip trailing " Equity"
    if ticker.endswith(" Equity"):
        ticker = ticker[: -len(" Equity")].strip()

    # Strip exchange suffix (last token if it matches known suffixes)
    parts = ticker.rsplit(" ", 1)
    if len(parts) == 2 and parts[1] in _EXCHANGE_SUFFIXES:
        ticker = parts[0]

    return ticker


def _cumulative_split_factor(dates, eff_dates, ratios: np.ndarray | list) -> np.ndarray:
    """Per-date cumulative forward-split factor for back-adjusting raw prices.

    ``factor(d) = product of ratios[i] for every split whose effective date is
    STRICTLY AFTER d`` (a quote ON the effective date already reflects the new,
    post-split strikes). Dividing a RAW strike / premium by this factor maps it
    into the split-adjusted frame Bloomberg OHLCV uses, so the option-premium
    rail and the engine's spot/strike live in the same scale. Returns all-ones
    when there are no splits. Vectorized (suffix-product + searchsorted).
    """
    d = pd.to_datetime(pd.Series(dates), errors="coerce").to_numpy("datetime64[ns]")
    if eff_dates is None or len(eff_dates) == 0:
        return np.ones(len(d))
    eff = pd.to_datetime(pd.Series(eff_dates), errors="coerce").to_numpy("datetime64[ns]")
    rat = np.asarray(ratios, dtype=float)
    order = np.argsort(eff)
    eff = eff[order]
    rat = rat[order]
    # suffix[i] = product(rat[i:]); suffix[len] = 1.0
    suffix = np.ones(len(rat) + 1)
    suffix[:-1] = np.cumprod(rat[::-1])[::-1]
    idx = np.searchsorted(eff, d, side="right")  # count of eff_dates <= d
    return suffix[idx]


class MarketDataConnector:
    """Unified data layer for Smart Wheel Engine.

    Loads, normalizes, and serves Bloomberg data to all engine modules.
    Handles ticker normalization (Bloomberg format -> standard),
    date parsing, and provides query methods.

    All date parameters accept ``"YYYY-MM-DD"`` strings.  Internally
    they are converted to ``pd.Timestamp`` for filtering.

    DataFrames are lazy-loaded on first access and cached for the
    lifetime of the connector instance.  If a CSV file is missing,
    methods return empty DataFrames or ``None`` rather than raising.
    """

    # CSV file names expected under ``data_dir``
    _FILES = {
        "ohlcv": "sp500_ohlcv.csv",
        "vol_iv": "sp500_vol_iv_full.csv",
        "dividends": "sp500_dividends.csv",
        "earnings": "sp500_earnings.csv",
        "treasury": "treasury_yields.csv",
        "vix": "vix_term_structure.csv",
        "fundamentals": "sp500_fundamentals.csv",
        "credit_risk": "sp500_credit_risk.csv",
        "liquidity": "sp500_liquidity.csv",
        "corporate_actions": "sp500_corporate_actions.csv",
    }

    # Deep-history slice manifest (R2 / docs/DATA_LAYER_DEEP_READ_DESIGN.md §A1).
    # Per series, the deep + delisted gz panels (under ``data_dir``) that extend
    # the recent monolith back to the 1994/1990 floor and add the ~1,015 delisted
    # constituents. Read ONLY when ``deep_history`` is enabled (default OFF). The
    # order matters: slices are concatenated AFTER the recent monolith and
    # de-duplicated keep-first, so precedence is
    # ``recent monolith > deep-current > delisted`` (a relisted held name's live
    # rows always win over a stale delisted row on the same (ticker, date)).
    # A manifest (vs. hardcoded names) lets future shards register without
    # touching ``_load``. A missing slice file is logged and skipped, never
    # raised — matching the connector's "missing CSV degrades, never crashes"
    # contract, so the feature is safe to ship before the gz exist everywhere.
    _DEEP_SLICES: dict[str, tuple[str, ...]] = {
        "ohlcv": (
            "deep/sp500_ohlcv__1994_2018.csv.gz",
            "deep/sp500_ohlcv__delisted.csv.gz",
        ),
        "vol_iv": (
            "deep/sp500_vol_iv_full__1994_2012.csv.gz",
            "deep/sp500_vol_iv_full__2012_2018.csv.gz",
            "deep/sp500_vol_iv__delisted.csv.gz",
        ),
        "liquidity": (
            "deep/sp500_liquidity__1994_2015.csv.gz",
            "deep/sp500_liquidity__delisted.csv.gz",
        ),
    }

    # Deep-IV sentinel floor (R7). The deep vol_iv panels carry a corrupt
    # implied-vol sentinel of magnitude ~134217.7 (≈ 2**27/1000), confined to
    # 1994-95 + a few delisted names (the 2026-06-05 QA's CONCERN-2). Left in, it
    # poisons IV-rank / z-score / VRP. On the assembled (deep) vol_iv read we NULL
    # ``hist_put_imp_vol`` / ``hist_call_imp_vol`` ABOVE this floor (keeping the
    # row — realized-vol columns may still be valid). The floor is set at 10000
    # (10,000% implied vol — physically absurd), NOT the ~500% the early design
    # note suggested: inspecting the delisted panel on the bytes showed real
    # distressed-name implied vols of 500-1196% that a 500 cut would wrongly
    # discard, while the sentinel sits alone at ~134217.7 with a clean gap below.
    _DEEP_IV_SENTINEL_FLOOR: float = 10_000.0
    _DEEP_IV_COLS: tuple[str, ...] = ("hist_put_imp_vol", "hist_call_imp_vol")

    # Low-end IV floor (R7 / audit W1+W8). vol_iv implied-vol is authoritatively
    # in PERCENT (e.g. 30.79 means 30.79%); a reading at or below 3.0 (= 3% vol)
    # is implausibly low for any real equity — the W8 low-tail garbage (e.g. the
    # 0.01 cells). We NULL such ``hist_put_imp_vol`` / ``hist_call_imp_vol`` cells
    # on EVERY served vol_iv read (monolith + assembled), keeping the row so the
    # realized-vol columns survive. Setting the floor exactly at the rankers' /
    # tracker's percent->decimal threshold (``if iv > 3.0: iv/100`` in
    # wheel_runner + wheel_tracker) means every IV the engine ever sees is
    # unambiguously percent (> 3.0), so that conditional conversion is ALWAYS
    # correct — this is the data-layer fix that obviates #356 (W1) and #360
    # WITHOUT a decision-trio edit. Realized-vol columns (``volatility_*``) are
    # left untouched: a low realized vol is legitimate and feeds F4 / VRP.
    _IV_LOW_FLOOR: float = 3.0

    def __init__(
        self, data_dir: str = "data/bloomberg", *, deep_history: bool | None = None
    ) -> None:
        self._data_dir = Path(data_dir)
        # Deep-history assembly (R2). DEFAULT OFF — the recent-monolith fast path
        # is unchanged until an architect-reviewed re-baseline flips it on (it is
        # a re-baseline event: it changes what EVEngine sees). When ``None`` the
        # env var ``SWE_DEEP_HISTORY`` (1/true/yes/on) decides; an explicit
        # bool argument always wins. Keyword-only so the positional ``data_dir``
        # contract every existing caller relies on is untouched.
        if deep_history is None:
            deep_history = os.environ.get("SWE_DEEP_HISTORY", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self._deep_history = bool(deep_history)
        # Cache for loaded DataFrames – keys match ``_FILES`` keys
        self._cache: dict[str, pd.DataFrame] = {}
        # Per-DataFrame ``{ticker: sub-frame}`` index, built lazily on the
        # first ``_filter_ticker`` call against a given cached frame and keyed
        # by ``id(df)``. Collapses the repeated O(rows) object-column scan a
        # universe sweep would otherwise do (one full pass instead of one per
        # ticker). Keyed by id() is safe because the frames live in
        # ``self._cache`` for the connector's lifetime, so their identity is
        # stable and they are never GC'd out from under the key.
        self._ticker_groups: dict[int, dict[str, pd.DataFrame]] = {}
        # Option-premium rail (Phase-2 prep). Per-ticker EOD-mid parquets
        # produced by ``scripts/produce_option_premiums.py`` from the Theta
        # option-history larder, served by ``get_option_premium*`` below.
        # Resolved from the module path (override with ``SWE_OPTION_PREMIUM_DIR``
        # in tests) and NOT under ``data_dir`` / ``_FILES`` — the files live in
        # the gitignored ``data_processed/`` tree, so they never enter
        # ``connector_data_sha256`` (no re-baseline) and the accessor degrades to
        # an empty frame (→ synthetic-BSM fallback) wherever they are absent
        # (CI, fresh clones). Cached per ticker for the connector's lifetime.
        _optprem_env = os.environ.get("SWE_OPTION_PREMIUM_DIR", "").strip()
        self._option_premium_dir = (
            Path(_optprem_env)
            if _optprem_env
            else Path(__file__).resolve().parent.parent / "data_processed" / "option_premium"
        )
        self._option_premium_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, key: str) -> pd.DataFrame:
        """Lazy-load a CSV by key, normalize tickers, cache the result.

        Default path (``deep_history`` OFF) is unchanged: read the single recent
        monolith. With ``deep_history`` ON and ``key`` in :attr:`_DEEP_SLICES`,
        delegate to :meth:`_load_assembled`, which extends the monolith with the
        deep + delisted slices BELOW this method — so every ``get_*`` accessor
        and the ranker above see longer history through unchanged signatures
        (CLAUDE.md §2: the decision-layer trio is untouched).
        """
        if key in self._cache:
            return self._cache[key]

        # Deep-history assembly is opt-in and only for the three per-name panels.
        if self._deep_history and key in self._DEEP_SLICES:
            return self._load_assembled(key)

        path = self._data_dir / self._FILES[key]
        if not path.exists():
            logger.warning("Data file not found: %s", path)
            self._cache[key] = pd.DataFrame()
            return self._cache[key]

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            logger.exception("Failed to read %s", path)
            self._cache[key] = pd.DataFrame()
            return self._cache[key]

        # Normalize ticker column if present. Map over the ~500 UNIQUE raw
        # tickers rather than ``.apply`` over every row — same values, but it
        # turns ~2.4M per-row ``normalize_ticker`` calls (OHLCV + IV combined)
        # into ~500. Output is identical to the prior ``.apply``.
        if "ticker" in df.columns:
            _norm = {t: normalize_ticker(t) for t in df["ticker"].unique()}
            df["ticker"] = df["ticker"].map(_norm)

        # Parse common date columns
        for col in (
            "date",
            "ex_date",
            "declared_date",
            "record_date",
            "payable_date",
            "announcement_date",
        ):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # R7/W8: clean the vol_iv implied-vol tails on the monolith read too
        # (the assembled deep read cleans via the same helper). Default-path fix.
        if key == "vol_iv":
            self._clean_vol_iv_inplace(df)

        self._cache[key] = df
        logger.info("Loaded %s: %d rows from %s", key, len(df), path)
        return df

    def _load_assembled(self, key: str) -> pd.DataFrame:
        """Assemble recent monolith ∪ deep ∪ delisted for a per-name panel.

        Concatenates the recent monolith FIRST, then each present deep/delisted
        slice from :attr:`_DEEP_SLICES`; normalizes tickers + parses dates over
        the combined frame; de-duplicates ``(ticker, date)`` keep-first (so
        precedence is recent > deep-current > delisted — a relisted held name's
        live rows win over a stale delisted row on a shared key); and sorts.
        Missing slices are logged and skipped (degrade to whatever is present).
        Cached for the connector lifetime exactly like :meth:`_load`.
        """
        parts: list[pd.DataFrame] = []

        recent_path = self._data_dir / self._FILES[key]
        if recent_path.exists():
            try:
                parts.append(pd.read_csv(recent_path, low_memory=False))
            except Exception:
                logger.exception("Failed to read %s", recent_path)
        else:
            logger.warning("Recent monolith not found: %s", recent_path)

        for rel in self._DEEP_SLICES[key]:
            slice_path = self._data_dir / rel
            if not slice_path.exists():
                logger.warning("Deep slice not found, skipping: %s", slice_path)
                continue
            try:
                comp = "gzip" if slice_path.suffix == ".gz" else None
                parts.append(pd.read_csv(slice_path, compression=comp, low_memory=False))
            except Exception:
                logger.exception("Failed to read deep slice %s", slice_path)

        if not parts:
            logger.warning("No sources found for assembled key %r", key)
            self._cache[key] = pd.DataFrame()
            return self._cache[key]

        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

        # Normalize tickers over the combined frame (map over uniques — identical
        # values to a per-row apply, far fewer calls).
        if "ticker" in df.columns:
            _norm = {t: normalize_ticker(t) for t in df["ticker"].unique()}
            df["ticker"] = df["ticker"].map(_norm)

        # Parse common date columns (same set as _load).
        for col in (
            "date",
            "ex_date",
            "declared_date",
            "record_date",
            "payable_date",
            "announcement_date",
        ):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # R7/W8: clean the vol_iv implied-vol tails (high ~134217.7 sentinel +
        # low sub-3% garbage), keeping the row. Same helper as the monolith read
        # so both paths share one clean PERCENT band.
        if key == "vol_iv":
            self._clean_vol_iv_inplace(df)

        # Dedup keep-first (precedence = concat order = recent > deep > delisted)
        # then sort. Guards the ~90 relisted held names so their stale delisted
        # rows never shadow the live monolith on a shared (ticker, date).
        if "ticker" in df.columns and "date" in df.columns:
            df = (
                df.drop_duplicates(subset=["ticker", "date"], keep="first")
                .sort_values(["ticker", "date"])
                .reset_index(drop=True)
            )

        self._cache[key] = df
        logger.info(
            "Loaded %s (deep_history): %d rows from %d source(s)",
            key,
            len(df),
            len(parts),
        )
        return df

    def _clean_vol_iv_inplace(self, df: pd.DataFrame) -> None:
        """NULL corrupt implied-vol cells on a vol_iv frame, keeping the row.

        Applied to BOTH the monolith (:meth:`_load`) and the assembled deep
        read (:meth:`_load_assembled`) so every IV consumer sees a clean PERCENT
        band ``(_IV_LOW_FLOOR, _DEEP_IV_SENTINEL_FLOOR]``: the high cut removes
        the ~134217.7 deep sentinel; the low cut removes the W8 sub-3% garbage
        (and makes the downstream ``if iv > 3.0`` percent->decimal conversion in
        wheel_runner / wheel_tracker always correct — #356 / #360 — with no
        decision-trio edit). Only ``hist_put_imp_vol`` / ``hist_call_imp_vol``
        are touched; realized-vol columns (``volatility_*``) are preserved.
        """
        for col in self._DEEP_IV_COLS:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                df[col] = numeric.where(
                    (numeric > self._IV_LOW_FLOOR) & (numeric <= self._DEEP_IV_SENTINEL_FLOOR)
                )

    def _clean_served_iv(self, value: object) -> float | None:
        """Apply the #363 IV band to a single served implied-vol cell.

        #369 / W27: the frame-level #363 gate (:meth:`_clean_vol_iv_inplace`)
        runs ONLY for ``key=='vol_iv'`` — i.e. the PIT ``get_iv_history`` path.
        The fundamentals snapshot IV (``implied_vol_atm``, which the put / CC /
        strangle rankers use as their fallback when ``get_iv_history`` is empty
        or absent) bypassed it and relied solely on the consumers' inline
        ``if iv > 3.0: iv /= 100`` heuristic. That heuristic silently accepted a
        sub-3 garbage reading (e.g. ``2.0``) as a 200 % *decimal* IV instead of
        rejecting it. Cleaning the served value here NULLs exactly the
        out-of-band cells the vol_iv gate would (``(_IV_LOW_FLOOR,
        _DEEP_IV_SENTINEL_FLOOR]``), so the fallback IV is unambiguously PERCENT
        (> 3.0) for every consumer and the downstream percent->decimal
        conversion is always correct — with no decision-trio edit (the #363
        precedent). Returns the value as a float when in-band, ``None`` when the
        cell is ``None``, else ``nan`` (NULL — degenerate/sentinel).
        """
        if value is None:
            return None
        try:
            iv = float(value)
        except (TypeError, ValueError):
            return float("nan")
        if not (self._IV_LOW_FLOOR < iv <= self._DEEP_IV_SENTINEL_FLOOR):
            return float("nan")
        return iv

    @staticmethod
    def _to_ts(value: str | None) -> pd.Timestamp | None:
        """Convert an optional date string to Timestamp."""
        if value is None:
            return None
        return pd.Timestamp(value)

    @staticmethod
    def _filter_dates(
        df: pd.DataFrame,
        date_col: str,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        """Filter a DataFrame by a date range (inclusive)."""
        if df.empty:
            return df
        out = df
        if start is not None:
            out = out[out[date_col] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out[date_col] <= pd.Timestamp(end)]
        return out

    def _filter_ticker(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Filter DataFrame to a single (already-normalized) ticker.

        Output-identical to ``df[df["ticker"] == ticker]`` (same rows, same
        original order, same columns; empty frame with the same schema when the
        ticker is absent). The speed-up is the lazily-built, cached
        ``{ticker: sub-frame}`` index (``self._ticker_groups``): the first call
        against a frame does a single ``groupby`` pass, every subsequent
        per-ticker lookup is O(1) — so a full-universe sweep makes one pass over
        each data file instead of one object-column scan per ticker.
        """
        if df.empty or "ticker" not in df.columns:
            return df
        key = id(df)
        groups = self._ticker_groups.get(key)
        if groups is None:
            # ``groupby`` preserves within-group row order, matching the
            # boolean-mask semantics this replaces. NB: must iterate the
            # (name, group) pairs explicitly — ``dict(df.groupby(...))`` is NOT
            # equivalent (GroupBy exposes ``.keys``, so ``dict()`` takes the
            # mapping-protocol path and builds the wrong thing). Hence noqa C416.
            groups = {t: sub for t, sub in df.groupby("ticker", sort=False)}  # noqa: C416
            self._ticker_groups[key] = groups
        hit = groups.get(ticker)
        if hit is None:
            return df.iloc[0:0]  # empty, same columns/dtypes — matches mask miss
        return hit

    # ------------------------------------------------------------------
    # Ticker normalization (public static)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_ticker(bbg_ticker: str) -> str:
        """Convert ``'AAPL UW Equity'`` or ``'AAPL UW'`` -> ``'AAPL'``."""
        return normalize_ticker(bbg_ticker)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data for *ticker*.

        Returns a DataFrame with a ``DatetimeIndex`` named ``date`` and
        columns ``open, high, low, close, volume``.

        AUDIT-VIII P1.5: the Bloomberg CSV ships with column labels
        rotated one position (``open=HIGH, high=CLOSE, close=OPEN,
        low=LOW``). We rename to the correct labels then verify the
        rename is still load-bearing: if the CSV is ever regenerated
        in the canonical order, the rename would silently invert the
        entire engine's view of price history. The post-rename
        invariant ``high >= max(open, close, low)`` MUST hold on a
        sampled set of rows; otherwise we log a critical warning.
        """
        df = self._load("ohlcv")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "date", start_date, end_date)
        if df.empty:
            return df
        df = df.rename(
            columns={
                "open": "high",
                "high": "close",
                "close": "open",
            }
        )
        out = (
            df[["date", "open", "high", "low", "close", "volume"]]
            .sort_values("date")
            .set_index("date")
        )
        self._validate_ohlcv_invariants(ticker, out)
        return out

    _ohlcv_invariant_warned: bool = False

    def _validate_ohlcv_invariants(self, ticker: str, df: pd.DataFrame) -> None:
        """Sample-check that ``high >= max(o,c,l)`` and ``low <= min(o,c,h)``.

        Runs at most once per connector instance (first OHLCV fetch)
        and emits a CRITICAL log line if the rename is no longer
        load-bearing. We never raise — the engine must degrade rather
        than hard-fail on a connector quirk — but the warning is
        unmistakable in any downstream log scraper.
        """
        if MarketDataConnector._ohlcv_invariant_warned:
            return
        try:
            sample = df.dropna(subset=["open", "high", "low", "close"]).tail(50)
            if sample.empty:
                return
            bad_high = (sample["high"] < sample[["open", "close", "low"]].max(axis=1)).sum()
            bad_low = (sample["low"] > sample[["open", "close", "high"]].min(axis=1)).sum()
            if bad_high + bad_low > 5:
                logger.critical(
                    "OHLCV invariant violation for %s: high<max(o,c,l) in %d rows, "
                    "low>min(o,c,h) in %d rows (of 50 sampled). The Bloomberg "
                    "CSV column-rename assumption may have drifted. EV / "
                    "feature math is at risk. Inspect data/bloomberg/sp500_ohlcv.csv.",
                    ticker,
                    bad_high,
                    bad_low,
                )
                MarketDataConnector._ohlcv_invariant_warned = True
        except Exception:
            # Never crash the engine on a diagnostic check.
            pass

    # ------------------------------------------------------------------
    # Volatility & IV
    # ------------------------------------------------------------------

    def get_iv_history(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get historical IV data for *ticker*.

        Columns: ``hist_put_imp_vol, hist_call_imp_vol,
        volatility_30d, volatility_60d, volatility_90d, volatility_260d``.
        """
        df = self._load("vol_iv")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "date", start_date, end_date)
        if df.empty:
            return df
        value_cols = [
            "date",
            "hist_put_imp_vol",
            "hist_call_imp_vol",
            "volatility_30d",
            "volatility_60d",
            "volatility_90d",
            "volatility_260d",
        ]
        present = [c for c in value_cols if c in df.columns]
        return df[present].sort_values("date").set_index("date")

    def _iv_series(self, ticker: str, as_of: str | None, lookback_days: int) -> pd.Series:
        """Return the 30-day IV series for *ticker* up to *as_of*."""
        end = as_of
        if as_of is not None:
            start = str(pd.Timestamp(as_of) - pd.Timedelta(days=int(lookback_days * 1.6)))[:10]
        else:
            start = None
        iv = self.get_iv_history(ticker, start_date=start, end_date=end)
        if iv.empty or "hist_put_imp_vol" not in iv.columns:
            return pd.Series(dtype=float)
        # Use average of put and call IV as the composite IV
        series = (iv["hist_put_imp_vol"] + iv["hist_call_imp_vol"]) / 2.0
        series = series.dropna()
        # Trim to exact lookback window
        if as_of is not None:
            cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)
            series = series[series.index >= cutoff]
        else:
            series = series.iloc[-lookback_days:]
        return series

    def get_iv_rank(
        self,
        ticker: str,
        as_of: str | None = None,
        lookback_days: int = 252,
    ) -> float:
        """IV Rank: ``(current - min) / (max - min)`` over *lookback_days*.

        Returns a value in ``[0, 1]`` or ``NaN`` if insufficient data.
        """
        series = self._iv_series(ticker, as_of, lookback_days)
        if series.empty:
            return float("nan")
        current = series.iloc[-1]
        lo, hi = series.min(), series.max()
        if hi == lo:
            return 0.5
        return float((current - lo) / (hi - lo))

    def get_iv_percentile(
        self,
        ticker: str,
        as_of: str | None = None,
        lookback_days: int = 252,
    ) -> float:
        """IV Percentile: fraction of days with IV below the current level.

        Returns a value in ``[0, 1]`` or ``NaN`` if insufficient data.
        """
        series = self._iv_series(ticker, as_of, lookback_days)
        if series.empty:
            return float("nan")
        current = series.iloc[-1]
        return float((series < current).sum() / len(series))

    def get_vol_risk_premium(self, ticker: str, as_of: str | None = None) -> float:
        """Volatility risk premium: IV - RV (30-day windows).

        Positive values indicate that implied vol exceeds realized vol,
        which is the typical premium harvested by option sellers.
        """
        end = as_of
        iv = self.get_iv_history(ticker, end_date=end)
        if iv.empty:
            return float("nan")
        last = iv.iloc[-1]
        impl = np.nanmean(
            [last.get("hist_put_imp_vol", np.nan), last.get("hist_call_imp_vol", np.nan)]
        )
        realized = last.get("volatility_30d", np.nan)
        if np.isnan(impl) or np.isnan(realized):
            return float("nan")
        return float(impl - realized)

    # ------------------------------------------------------------------
    # Events – Earnings
    # ------------------------------------------------------------------

    def get_earnings(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get earnings announcements for *ticker*.

        Columns: ``year_period, announcement_date, announcement_time,
        earnings_eps, comparable_eps, estimate_eps``.
        """
        df = self._load("earnings")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "announcement_date", start_date, end_date)
        if df.empty:
            return df
        rename = {}
        if "year/period" in df.columns:
            rename["year/period"] = "year_period"
        out = df.rename(columns=rename).sort_values("announcement_date")
        cols = [
            c
            for c in [
                "year_period",
                "announcement_date",
                "announcement_time",
                "earnings_eps",
                "comparable_eps",
                "estimate_eps",
            ]
            if c in out.columns
        ]
        return out[cols].reset_index(drop=True)

    def get_next_earnings(self, ticker: str, as_of: str | None = None) -> dict | None:
        """Return the next upcoming earnings event after *as_of*.

        Returns a dict with keys ``announcement_date``,
        ``announcement_time``, ``estimate_eps``, and ``year_period``
        or ``None`` if no future earnings are found.
        """
        ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
        df = self._load("earnings")
        df = self._filter_ticker(df, ticker)
        if df.empty or "announcement_date" not in df.columns:
            return None
        future = df[df["announcement_date"] > ref].sort_values("announcement_date")
        if future.empty:
            return None
        row = future.iloc[0]
        return {
            "announcement_date": row["announcement_date"],
            "announcement_time": row.get("announcement_time"),
            "estimate_eps": row.get("estimate_eps"),
            "year_period": row.get("year/period", row.get("year_period")),
        }

    def get_recent_earnings(
        self,
        ticker: str,
        as_of: str | None = None,
        lookback_days: int = 7,
    ) -> dict | None:
        """Return the most recent earnings event in ``[as_of - lookback_days, as_of]``.

        The symmetric complement of :meth:`get_next_earnings` — where
        that returns the next FUTURE earnings, this returns the most
        recent PAST earnings within a lookback window. Both are needed
        by :class:`~engine.event_gate.EventGate`, whose
        ``_event_touches_window`` arithmetic
        (``window_start = trade_start - timedelta(days=buf)``) and
        ``±{buf}d buffer`` reason string make the gate explicitly
        symmetric (S23 F1). Without this method, callers register only
        forward earnings; a trade opened just after an earnings
        announcement is silently allowed even though the post-event
        IV-crush / drift window is exactly what the gate's docstring
        cites as the motivation for blocking.

        Returns a dict with the same keys as :meth:`get_next_earnings`
        (``announcement_date``, ``announcement_time``, ``estimate_eps``,
        ``year_period``) or ``None`` if no past earnings are found in
        the window. The two methods are *complementary* — the
        ``> ref`` / ``<= ref`` cutoff is set so an event ON ``as_of``
        is treated as past (returned by this method, not by
        ``get_next_earnings``).
        """
        ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
        lookback_start = ref - pd.Timedelta(days=int(lookback_days))
        df = self._load("earnings")
        df = self._filter_ticker(df, ticker)
        if df.empty or "announcement_date" not in df.columns:
            return None
        past = df[
            (df["announcement_date"] >= lookback_start) & (df["announcement_date"] <= ref)
        ].sort_values("announcement_date")
        if past.empty:
            return None
        row = past.iloc[-1]  # most recent within the lookback
        return {
            "announcement_date": row["announcement_date"],
            "announcement_time": row.get("announcement_time"),
            "estimate_eps": row.get("estimate_eps"),
            "year_period": row.get("year/period", row.get("year_period")),
        }

    # ------------------------------------------------------------------
    # Events – Corporate actions
    # ------------------------------------------------------------------

    def get_corporate_actions(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of: str | None = None,
        include_regular_cash: bool = False,
    ) -> pd.DataFrame:
        """Corporate actions for *ticker*, filtered on ``effective_date``.

        Columns: ``announcement_date, effective_date, action_type, ratio,
        amount`` (subset present). Sorted by ``effective_date``.

        By default the ~94 %-majority ``"Regular Cash"`` rows (ordinary
        dividends — already handled by the dividend / ex-div path, and far too
        frequent to gate on) are **excluded**, so what remains is the set of
        *disruptive* actions (splits, spinoffs, split-offs, special cash, rights,
        return of capital, …) the wheel's empirical forward distribution does
        not model. This is the data source for the #3A corporate-action event
        lockout (``engine.event_gate``, ``kind="corp_action"``). Pass
        ``include_regular_cash=True`` for the raw set.

        Point-in-time: when ``as_of`` is supplied, rows whose
        ``announcement_date`` is **after** ``as_of`` are dropped — a backtest at
        ``as_of`` must only see corporate actions that were already *announced*
        by then, even if their ``effective_date`` is still in the future. This
        keeps the event-gate registration lookahead-free.
        """
        df = self._load("corporate_actions")
        df = self._filter_ticker(df, ticker)
        if df.empty:
            return df
        # The corporate-actions CSV ships its date columns as strings (unlike
        # the date-parsed monoliths); parse on a COPY so the cached frame is not
        # mutated, then range/PIT-filter.
        df = df.copy()
        for _dc in ("effective_date", "announcement_date"):
            if _dc in df.columns:
                df[_dc] = pd.to_datetime(df[_dc], errors="coerce")
        df = self._filter_dates(df, "effective_date", start_date, end_date)
        if df.empty:
            return df
        if as_of is not None and "announcement_date" in df.columns:
            df = df[df["announcement_date"] <= pd.Timestamp(as_of)]
            if df.empty:
                return df
        if not include_regular_cash and "action_type" in df.columns:
            df = df[df["action_type"].astype(str).str.strip() != "Regular Cash"]
        cols = [
            c
            for c in [
                "announcement_date",
                "effective_date",
                "action_type",
                "ratio",
                "amount",
            ]
            if c in df.columns
        ]
        return df[cols].sort_values("effective_date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Options – real EOD premiums (Phase-2 prep rail)
    # ------------------------------------------------------------------

    def _load_option_premium(self, ticker: str) -> pd.DataFrame:
        """Lazy-load the per-ticker option-premium parquet, cached.

        Reads ``<option_premium_dir>/<TICKER>.parquet`` (produced by
        ``scripts/produce_option_premiums.py`` from the Theta option-history
        larder). Returns an EMPTY frame — never raises — when the file is
        absent or unreadable, so every accessor below degrades to "no real
        premium" and callers fall back to the synthetic-BSM path. The frame is
        cached for the connector's lifetime and never mutated after load (the
        accessors filter/copy), matching the ``_load`` cache invariant.
        """
        t = normalize_ticker(ticker)
        if t in self._option_premium_cache:
            return self._option_premium_cache[t]

        path = self._option_premium_dir / f"{t}.parquet"
        if not path.exists():
            self._option_premium_cache[t] = pd.DataFrame(columns=OPTION_PREMIUM_COLUMNS)
            return self._option_premium_cache[t]
        try:
            df = pd.read_parquet(path)
        except Exception:
            logger.exception("Failed to read option-premium parquet %s", path)
            self._option_premium_cache[t] = pd.DataFrame(columns=OPTION_PREMIUM_COLUMNS)
            return self._option_premium_cache[t]

        # A present-but-malformed file (missing a contract column) must degrade
        # like an absent one — the accessors below filter on ``expiration`` /
        # ``right`` / ``date``, so a parquet lacking any of them would KeyError
        # and crash the ranker, breaking the connector's "missing data degrades,
        # never crashes" contract. Validate the full column contract once here so
        # all three accessors stay raise-free.
        missing = [c for c in OPTION_PREMIUM_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(
                "option-premium parquet %s missing columns %s — ignoring file",
                path,
                missing,
            )
            self._option_premium_cache[t] = pd.DataFrame(columns=OPTION_PREMIUM_COLUMNS)
            return self._option_premium_cache[t]

        # Defensive: ensure the PIT date axes are datetime (parquet usually
        # round-trips them, but a hand-built / re-exported file may not).
        for _dc in ("date", "expiration"):
            if _dc in df.columns and not pd.api.types.is_datetime64_any_dtype(df[_dc]):
                df[_dc] = pd.to_datetime(df[_dc], errors="coerce")
        # Back-adjust RAW larder strikes/premiums into the engine's split-adjusted
        # frame (the larder is unadjusted; Bloomberg OHLCV is split-adjusted).
        df = self._split_adjust_option_premium(df, t)
        self._option_premium_cache[t] = df
        logger.info("Loaded option-premium: %d rows from %s", len(df), path)
        return df

    def _split_adjust_option_premium(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Back-adjust raw larder strikes/premiums into the split-adjusted frame.

        The Theta larder ships RAW (unadjusted) strikes and prices; the engine's
        OHLCV is Bloomberg SPLIT-adjusted. Without this, a name that split
        between the quote date and now (e.g. NVDA 10:1, eff 2024-06-10) would
        carry ~10x-off strikes vs the engine's spot, so the ranker's solved
        strike never matches a listed one (silent fallback) — or worse, matches a
        wrong one. Divide strike/bid/ask/mid/close by the cumulative forward
        ``Stock Split`` factor (``get_corporate_actions`` ratios). Degrades to
        unchanged when there is no split data or no split affects the window.
        Mutates the freshly-read (uncached) frame in place — safe, pre-cache.
        """
        if df.empty or "date" not in df.columns:
            return df
        try:
            ca = self.get_corporate_actions(ticker)
        except Exception:
            return df
        if ca.empty or "action_type" not in ca.columns or "effective_date" not in ca.columns:
            return df
        sp = ca[ca["action_type"].astype(str).str.strip() == "Stock Split"]
        if sp.empty or "ratio" not in sp.columns:
            return df
        ratios = pd.to_numeric(sp["ratio"], errors="coerce")
        keep = ratios.notna() & (ratios > 0)
        if not keep.any():
            return df
        factor = _cumulative_split_factor(
            df["date"], sp.loc[keep, "effective_date"].to_numpy(), ratios[keep].to_numpy()
        )
        # Keep the (common) no-split-in-window path numerically identical.
        if np.allclose(factor, 1.0):
            return df
        for col in ("strike", "bid", "ask", "mid", "close"):
            if col in df.columns:
                df[col] = df[col] / factor
        return df

    def get_option_premium_chain(
        self,
        ticker: str,
        expiry: str,
        as_of: str | None = None,
        *,
        max_staleness_days: int = 7,
    ) -> pd.DataFrame:
        """Real EOD option-premium chain for *ticker*'s ``expiry``, point-in-time.

        Returns one row per ``(strike, right)`` — columns
        :data:`OPTION_PREMIUM_COLUMNS` — for the single most-recent EOD snapshot
        whose ``date`` is ``<= as_of`` and within ``max_staleness_days`` of it
        (so a backtest at ``as_of`` never sees a future quote, and a long market
        holiday gap does not silently serve a stale market). With ``as_of=None``
        the latest available snapshot for that expiry is used, **bounded by the
        same ``max_staleness_days`` against today's wall clock** — ``as_of=None``
        means "the current market state", not "whatever the larder last saw", so
        a stale larder degrades to the synthetic-BSM fallback instead of pairing
        a weeks-old quote with a live caller (adversarial review 2026-07-01
        D1-1/AB-4: an unbounded latest-quote paired with a different-frontier
        spot inflated live EV 4-18x).

        ``mid = (bid + ask) / 2`` is the real premium. Returns an EMPTY frame
        (same columns) when no produced data exists / no PIT snapshot qualifies
        — callers fall back to the synthetic-BSM premium. Never raises.
        """
        df = self._load_option_premium(ticker)
        if df.empty:
            return df
        exp = pd.Timestamp(expiry).normalize()
        sub = df[df["expiration"] == exp]
        if sub.empty:
            return df.iloc[0:0]
        if as_of is not None:
            ref = pd.Timestamp(as_of).normalize()
            sub = sub[sub["date"] <= ref]
            if sub.empty:
                return df.iloc[0:0]
            snap = sub["date"].max()
            if (ref - snap).days > max_staleness_days:
                return df.iloc[0:0]
        else:
            snap = sub["date"].max()
            # D1-1/AB-4 hardening: bound the "latest" branch against today's
            # wall clock. Refuse-only — a quote this stale can only be replaced
            # by the synthetic-BSM fallback, never the other way around.
            if (pd.Timestamp.now().normalize() - snap).days > max_staleness_days:
                return df.iloc[0:0]
        sub = sub[sub["date"] == snap]
        return sub.sort_values(["right", "strike"]).reset_index(drop=True)

    def get_option_premium(
        self,
        ticker: str,
        expiry: str,
        strike: float,
        right: str,
        as_of: str | None = None,
        *,
        strike_tol: float | None = None,
        max_staleness_days: int = 7,
    ) -> dict | None:
        """One real EOD option premium for the listed strike nearest *strike*.

        ``right`` is ``"put"``/``"call"`` (``"p"``/``"c"`` accepted). Returns a
        dict of :data:`OPTION_PREMIUM_COLUMNS` for the nearest listed strike on
        the requested side as of ``as_of`` (PIT), or ``None`` when no produced
        data / PIT snapshot / matching side exists, or when the nearest strike is
        farther than ``strike_tol`` (when given). ``None`` is the caller's signal
        to use the synthetic-BSM premium — the wheel rankers delta-solve a
        continuous strike, so they snap to the nearest *listed* strike here.
        """
        chain = self.get_option_premium_chain(
            ticker, expiry, as_of, max_staleness_days=max_staleness_days
        )
        if chain.empty:
            return None
        r = str(right).strip().lower()
        r = {"p": "put", "c": "call"}.get(r, r)
        side = chain[chain["right"] == r]
        if side.empty:
            return None
        diffs = (side["strike"] - float(strike)).abs()
        idx = diffs.idxmin()
        if strike_tol is not None and float(diffs.loc[idx]) > float(strike_tol):
            return None
        row = side.loc[idx]
        return {c: row[c] for c in OPTION_PREMIUM_COLUMNS if c in side.columns}

    def list_option_expirations(
        self,
        ticker: str,
        as_of: str | None = None,
        *,
        min_dte: int | None = None,
        max_dte: int | None = None,
    ) -> list[pd.Timestamp]:
        """Sorted listed expirations with a real EOD snapshot available PIT.

        Only expirations that have at least one snapshot ``date <= as_of`` are
        returned (a backtest at ``as_of`` must not "see" a contract that had not
        yet quoted). ``min_dte`` / ``max_dte`` (relative to ``as_of``, or to the
        latest snapshot date when ``as_of`` is None) narrow to the wheel's DTE
        belt so the ranker can snap a DTE target to a listed expiry. Empty list
        when no produced data exists, or — mirroring
        :meth:`get_option_premium_chain` — when ``as_of`` is ``None`` and the
        larder's freshest snapshot is more than 7 days behind today's wall
        clock (``as_of=None`` means "the current market state"; D1-1/AB-4).
        """
        df = self._load_option_premium(ticker)
        if df.empty:
            return []
        sub = df
        ref = pd.Timestamp(as_of).normalize() if as_of is not None else None
        if ref is not None:
            sub = sub[sub["date"] <= ref]
        if sub.empty:
            return []
        if ref is None:
            ref = sub["date"].max()
            # D1-1/AB-4 hardening — same wall-clock bound as
            # get_option_premium_chain's as_of=None branch (refuse-only).
            if (pd.Timestamp.now().normalize() - ref).days > 7:
                return []
        exps = sorted(pd.Timestamp(e) for e in sub["expiration"].dropna().unique())
        out: list[pd.Timestamp] = []
        for e in exps:
            dte = (e - ref).days
            if min_dte is not None and dte < min_dte:
                continue
            if max_dte is not None and dte > max_dte:
                continue
            out.append(e)
        return out

    # ------------------------------------------------------------------
    # Events – Dividends
    # ------------------------------------------------------------------

    def get_dividends(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get dividend history for *ticker*.

        Filtering uses the ``ex_date`` column.  Columns returned:
        ``declared_date, ex_date, record_date, payable_date,
        dividend_amount, dividend_frequency, dividend_type``.
        """
        df = self._load("dividends")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "ex_date", start_date, end_date)
        if df.empty:
            return df
        cols = [
            c
            for c in [
                "declared_date",
                "ex_date",
                "record_date",
                "payable_date",
                "dividend_amount",
                "dividend_frequency",
                "dividend_type",
            ]
            if c in df.columns
        ]
        return df[cols].sort_values("ex_date").reset_index(drop=True)

    def get_next_dividend(self, ticker: str, as_of: str | None = None) -> dict | None:
        """Return the next upcoming ex-dividend event after *as_of*.

        Returns a dict with ``ex_date``, ``dividend_amount``,
        ``payable_date``, ``dividend_frequency`` or ``None``.
        """
        ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
        df = self._load("dividends")
        df = self._filter_ticker(df, ticker)
        if df.empty or "ex_date" not in df.columns:
            return None
        future = df[df["ex_date"] > ref].sort_values("ex_date")
        if future.empty:
            return None
        row = future.iloc[0]
        return {
            "ex_date": row["ex_date"],
            "dividend_amount": row.get("dividend_amount"),
            "payable_date": row.get("payable_date"),
            "dividend_frequency": row.get("dividend_frequency"),
        }

    # ------------------------------------------------------------------
    # Rates
    # ------------------------------------------------------------------

    def get_risk_free_rate(self, as_of: str | None = None, tenor: str = "rate_3m") -> float:
        """Get risk-free rate from treasury yields.

        *tenor* must be one of ``rate_3m``, ``rate_6m``, ``rate_2y``,
        ``rate_10y``.  Returns the **decimal** rate (e.g. ``0.045``
        meaning 4.5%), matching the convention used by the rest of the
        engine (``EVEngine``, ``black_scholes_price``, HMM, etc.).

        AUDIT-VIII / D20: the raw treasury CSV is authoritatively in
        percent form (e.g. ``1.3757`` = 1.3757%, ``0.04`` = 0.04%), so we
        divide by 100 **unconditionally**. The previous value-based
        heuristic (``/100 only if > 1``) silently mis-read any sub-1%
        *percent* rate — e.g. a 0.04% ZIRP-era T-bill — as already-decimal,
        a 100x error across the entire 2011-2022 low-rate era (where ~56%
        of ``rate_3m`` rows are ≤ 1.0). Returns ``NaN`` if data is
        unavailable so callers can detect missing data explicitly rather
        than getting a default 0.
        """
        df = self._load("treasury")
        if df.empty or tenor not in df.columns:
            return float("nan")
        df = df.dropna(subset=[tenor])
        if as_of is not None:
            df = df[df["date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return float("nan")
        rate = float(df.sort_values("date").iloc[-1][tenor])
        if np.isnan(rate):
            return float("nan")
        return rate / 100.0

    # ------------------------------------------------------------------
    # VIX
    # ------------------------------------------------------------------

    def get_vix(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get VIX and term-structure history.

        Columns: ``vix, vix_3m, vix_6m`` with ``DatetimeIndex``.
        """
        df = self._load("vix")
        df = self._filter_dates(df, "date", start_date, end_date)
        if df.empty:
            return df
        cols = [c for c in ["date", "vix", "vix_3m", "vix_6m"] if c in df.columns]
        return df[cols].sort_values("date").set_index("date")

    def get_vix_regime(self, as_of: str | None = None) -> dict:
        """Characterize the VIX regime as of a given date.

        Returns a dict with:
        - ``vix``: current VIX level
        - ``vix_percentile``: percentile over full history (0-1)
        - ``term_structure``: ``'contango'`` or ``'backwardation'``
        - ``vix_3m``, ``vix_6m``: longer-dated VIX values
        """
        df = self._load("vix")
        if df.empty or "vix" not in df.columns:
            return {
                "vix": float("nan"),
                "vix_percentile": float("nan"),
                "term_structure": "unknown",
                "vix_3m": float("nan"),
                "vix_6m": float("nan"),
            }

        if as_of is not None:
            df = df[df["date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return {
                "vix": float("nan"),
                "vix_percentile": float("nan"),
                "term_structure": "unknown",
                "vix_3m": float("nan"),
                "vix_6m": float("nan"),
            }

        df = df.sort_values("date")
        last = df.iloc[-1]
        vix_val = float(last["vix"])
        all_vix = df["vix"].dropna()
        pct = float((all_vix < vix_val).sum() / len(all_vix)) if len(all_vix) > 0 else float("nan")
        vix_3m = float(last.get("vix_3m", np.nan))
        vix_6m = float(last.get("vix_6m", np.nan))

        if np.isnan(vix_3m):
            ts = "unknown"
        elif vix_val < vix_3m:
            ts = "contango"
        elif vix_val > vix_3m:
            ts = "backwardation"
        else:
            ts = "flat"

        return {
            "vix": vix_val,
            "vix_percentile": pct,
            "term_structure": ts,
            "vix_3m": vix_3m,
            "vix_6m": vix_6m,
        }

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(self, ticker: str, as_of: str | pd.Timestamp | None = None) -> dict | None:
        """Get a fundamentals snapshot for *ticker*.

        Returns a dict with keys such as ``pe_ratio``, ``beta``,
        ``market_cap``, ``sector``, ``industry_group``,
        ``dividend_yield``, ``fcf_yield``, ``roe``,
        ``debt_to_equity``, ``volatility_30d``, ``implied_vol_atm``,
        or ``None`` if the ticker is not found.

        Unit contract: ``dividend_yield``, ``volatility_30d`` and
        ``implied_vol_atm`` are passed through from the Bloomberg CSV in
        its native PERCENT units (e.g. ``2.04`` means 2.04%, ``26.17``
        means 26.17%). Callers feeding these into BSM / EV math must
        convert to decimals first — and must divide UNCONDITIONALLY,
        since a value below 1.0 is an ordinary sub-1% reading, not an
        already-decimal value.

        Point-in-time (``as_of``): when an ``as_of`` date is supplied, the
        ``dividend_yield`` (the BSM carry-``q`` source) is resolved
        **point-in-time** — the latest ``dvd_yld_12m`` on/before ``as_of``
        from the dated ``broad_pull/dividend_pit`` panel — instead of the
        current snapshot, closing the #354 / W-2 carry-``q`` lookahead. This
        is the connector half of Phase 3G; ONLY ``dividend_yield`` is made
        PIT here (the other fields remain the current snapshot pending broader
        PIT wiring). ``as_of=None`` (the default — and the ranked path today,
        since no consumer threads ``as_of`` yet) returns the snapshot unchanged,
        so the change is backward-compatible and **not EV-moving** until the
        supervised step threads ``as_of`` from ``wheel_runner``. The dated
        panel falls back to the snapshot when absent / NaN at ``as_of`` (never
        worse than today).
        """
        df = self._load("fundamentals")
        if df.empty:
            return None
        row = df[df["ticker"] == ticker]
        if row.empty:
            return None
        r = row.iloc[0]
        out = {
            "ticker": ticker,
            "pe_ratio": r.get("pe_ratio"),
            "best_pe_ratio": r.get("best_pe_ratio"),
            "beta": r.get("beta_raw_overridable"),
            "market_cap": r.get("cur_mkt_cap"),
            "dividend_yield": r.get("eqy_dvd_yld_12m"),
            "fcf_yield": r.get("free_cash_flow_yield"),
            "roe": r.get("return_com_eqy"),
            "debt_to_equity": r.get("tot_debt_to_tot_eqy"),
            "sector": r.get("gics_sector_name"),
            "industry_group": r.get("gics_industry_group_name"),
            "volatility_30d": r.get("volatility_30d"),
            "implied_vol_atm": self._clean_served_iv(r.get("30day_impvol_100.0%mny_df")),
        }
        if as_of is not None:
            pit_yield = self._pit_dividend_yield(ticker, as_of)
            if pit_yield is not None:
                out["dividend_yield"] = pit_yield
        return out

    def _load_dividend_pit_panel(self) -> pd.DataFrame | None:
        """The dated dividend-yield PIT panel (``broad_pull/dividend_pit``),
        loaded lazily and cached on the instance. ``None`` when the broad-pull
        data is absent (e.g. a fresh clone) — callers fall back to the snapshot.
        Loaded via ``BroadPullLoader`` (the canonical broad-pull reader; the
        same lazy ``engine -> data`` import pattern used in ``signal_context`` /
        ``wheel_runner``)."""
        if not hasattr(self, "_dividend_pit_panel"):
            try:
                from data.broad_pull_loaders import BroadPullLoader

                self._dividend_pit_panel = BroadPullLoader().load("dividend_pit")
            except Exception:
                self._dividend_pit_panel = None
        return self._dividend_pit_panel

    def _pit_dividend_yield(self, ticker: str, as_of: str | pd.Timestamp) -> float | None:
        """Latest dated ``dvd_yld_12m`` (PERCENT) on/before *as_of* for *ticker*
        from the dividend-PIT panel, or ``None`` if unavailable. Same PERCENT
        unit contract as the snapshot ``dividend_yield`` it overrides."""
        panel = self._load_dividend_pit_panel()
        if panel is None or panel.empty:
            return None
        key = normalize_ticker(ticker)
        sub = panel[
            (panel["ticker_normalized"] == key)
            & (panel["date"] <= pd.Timestamp(as_of))
            & (panel["dvd_yld_12m"].notna())
        ]
        if sub.empty:
            return None
        return float(sub.sort_values("date")["dvd_yld_12m"].iloc[-1])

    def get_credit_risk(self, ticker: str) -> dict | None:
        """Get credit risk metrics for *ticker*.

        Returns a dict with ``altman_z_score``,
        ``interest_coverage_ratio``, ``sp_rating`` or ``None``.
        """
        df = self._load("credit_risk")
        if df.empty:
            return None
        row = df[df["ticker"] == ticker]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "ticker": ticker,
            "altman_z_score": r.get("altman_z_score"),
            "interest_coverage_ratio": r.get("interest_coverage_ratio"),
            "sp_rating": r.get("rtg_sp_lt_lc_issuer_credit"),
        }

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self) -> list[str]:
        """Get all available tickers (normalized, deduplicated, sorted)."""
        tickers: set[str] = set()
        for key in ("ohlcv", "fundamentals", "vol_iv"):
            df = self._load(key)
            if not df.empty and "ticker" in df.columns:
                tickers.update(df["ticker"].dropna().unique())
        return sorted(tickers)

    def get_data_frontier(self, dataset: str = "ohlcv") -> pd.Timestamp | None:
        """Global data frontier — the max trade date across the whole table.

        Used by the EV rankers to resolve an ``as_of=None`` scan to a
        staleness reference: rather than letting each ticker independently
        resolve to its own latest bar (which silently admits index leavers
        like CTRA whose data ended 76+ days ago), the ranker compares each
        ticker's last bar against this global frontier and drops any ticker
        that is ``max_as_of_staleness_days`` behind.

        Clamps the raw ``_load`` max against ``date.today()`` so that a
        single future-dated corrupt row in any ticker cannot inflate the
        frontier and black out the entire universe at ``as_of=None``.  A
        corrupt-row frontier jump would make the failure mode "fewer
        candidates" (drop-only, never a rescue), and the clamp turns that
        into a no-op.  The clamped value is the correct semantic answer: the
        frontier cannot be later than today.

        Dataset-parameterized so a future supervised session can reuse the
        same helper for an IV frontier (``dataset='vol_iv'``, verified
        = 2026-06-04) without redesign.

        Note: ``engine_api._data_frontier`` probes AAPL via ``get_ohlcv``
        instead of reading ``_load`` directly so the HTTP status endpoint
        reflects what the connector actually filters.  Both return the same
        date today (2026-06-04), but they may diverge if a future OHLCV
        refresh lands a corrupt row that ``_load`` passes through but
        ``get_ohlcv``'s per-ticker filter would skip.  The clamp here
        mitigates the divergence risk for the staleness-gate use case.

        Returns ``None`` on any error or when the table is absent/empty —
        callers fall back to legacy (no staleness gate) via hasattr-guard.
        """
        import datetime as _dt

        try:
            df = self._load(dataset)
            if df.empty or "date" not in df.columns:
                return None
            raw_max = df["date"].max()
            if pd.isna(raw_max):
                return None
            frontier = pd.Timestamp(raw_max)
            # Clamp: the frontier cannot be later than today (corrupt future rows).
            today_ts = pd.Timestamp(_dt.date.today())
            if frontier > today_ts:
                frontier = today_ts
            return frontier
        except Exception:
            return None

    def screen_universe(
        self,
        min_market_cap: float = 0,
        max_pe: float | None = None,
        sectors: list[str] | None = None,
        min_iv_rank: float | None = None,
        max_beta: float | None = None,
        as_of: str | None = None,
    ) -> pd.DataFrame:
        """Screen the universe by fundamental and volatility criteria.

        Returns a DataFrame with one row per ticker that passes all
        filters.  Columns include ``ticker``, ``pe_ratio``, ``beta``,
        ``market_cap``, ``sector``, ``iv_rank`` (if requested).
        """
        df = self._load("fundamentals")
        if df.empty:
            return pd.DataFrame()

        result = df.copy()

        # Market cap filter
        if "cur_mkt_cap" in result.columns:
            result = result[result["cur_mkt_cap"].fillna(0) >= min_market_cap]

        # P/E filter
        if max_pe is not None and "pe_ratio" in result.columns:
            result = result[result["pe_ratio"].fillna(float("inf")) <= max_pe]

        # Beta filter
        if max_beta is not None and "beta_raw_overridable" in result.columns:
            result = result[result["beta_raw_overridable"].fillna(float("inf")) <= max_beta]

        # Sector filter
        if sectors is not None and "gics_sector_name" in result.columns:
            sectors_lower = {s.lower() for s in sectors}
            result = result[result["gics_sector_name"].str.lower().fillna("").isin(sectors_lower)]

        # Build output frame with friendly column names
        out_cols = {
            "ticker": "ticker",
            "pe_ratio": "pe_ratio",
            "beta_raw_overridable": "beta",
            "cur_mkt_cap": "market_cap",
            "gics_sector_name": "sector",
            "eqy_dvd_yld_12m": "dividend_yield",
            "volatility_30d": "volatility_30d",
        }
        out = result.rename(columns={k: v for k, v in out_cols.items() if k in result.columns})
        keep = [v for v in out_cols.values() if v in out.columns]
        out = out[keep].reset_index(drop=True)

        # IV rank filter (computed on-the-fly; potentially expensive).
        # AUDIT-VIII P1.3: ``as_of`` is threaded through so backtests do
        # not leak future IV rank. Live callers pass ``as_of=None`` and
        # get the current rank as before.
        if min_iv_rank is not None:
            iv_ranks = []
            for t in out["ticker"]:
                iv_ranks.append(self.get_iv_rank(t, as_of=as_of))
            out["iv_rank"] = iv_ranks
            out = out[out["iv_rank"].fillna(-1) >= min_iv_rank]
        elif "iv_rank" not in out.columns:
            # Don't add iv_rank column if not filtering by it
            pass

        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Liquidity (bonus – available data)
    # ------------------------------------------------------------------

    def get_liquidity(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get liquidity metrics (avg volume, turnover, shares out).

        Returns DataFrame with ``DatetimeIndex`` and columns
        ``avg_vol_30d, turnover, shares_out``.
        """
        df = self._load("liquidity")
        df = self._filter_ticker(df, ticker)
        df = self._filter_dates(df, "date", start_date, end_date)
        if df.empty:
            return df
        cols = [c for c in ["date", "avg_vol_30d", "turnover", "shares_out"] if c in df.columns]
        return df[cols].sort_values("date").set_index("date")
