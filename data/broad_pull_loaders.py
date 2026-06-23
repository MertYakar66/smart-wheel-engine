"""Broad-pull dataset loaders (Phase 0B — additive, dormant plumbing).

Read-only loaders for the ~25 net-new Bloomberg datasets pulled in the
2026-06-17/18 broad session and integrated under ``data/bloomberg/broad_pull/``
(see ``docs/WIRING_CAMPAIGN.md`` Phase 0B, ``docs/DATA_INVENTORY.md`` §6, and
``staging/BROAD_PULL_MANIFEST.md`` on branch
``claude/bloomberg-broad-pull-2026-06-17``).

§2 SAFETY — this module is PLAIN / not-EV-moving *by construction*:

* It is **not** imported or called by the decision trio (``ev_engine`` /
  ``wheel_runner`` / ``candidate_dossier``), any risk gate, any reviewer, or the
  production ``ConsolidatedBloombergLoader.load_all`` path. **Nothing consumes
  it.** Wiring any of these into a consumer (``skew_dynamics``, ``event_gate``,
  ``regime_detector``, ``option_pricer``, ``cost_model``, …) is the supervised,
  re-baseline-coupled Phase 1-3 work and is **out of scope** here.
* It only READS committed CSV/gz bytes into DataFrames; it changes no
  ``EVEngine.evaluate`` output.

The loaders mirror ``data/consolidated_loader.py``'s idiom (lowercase columns,
parse the date column, per-ticker access) with three additions the broad-pull
data needs:

* **gzip handling** — ``.csv.gz`` panels (``iv_surface``, ``vol_term_rv``) are
  read directly (pandas ``compression='infer'``);
* **float32 downcast** — the large numeric panels (millions of rows) are
  downcast to keep memory flat alongside other processes;
* **explicit winsorization** — the manifest's outlier-flagged columns are clipped
  to their 0.1%/99.9% quantiles and **every clamp is logged** (never silent).

Per-ticker access is *lazy*: the loader stores one panel DataFrame per dataset
and filters it on demand in :meth:`BroadPullLoader.series`, rather than eagerly
materialising a dict of ~510 per-ticker copies (which would double the memory of
the multi-million-row panels).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.consolidated_loader import normalize_ticker

logger = logging.getLogger(__name__)

# Connector-read location for the integrated broad-pull data (mirrors the
# staging/<bucket>/ structure on the broad-pull branch).
BROAD_PULL_DIR = Path("data/bloomberg/broad_pull")

# Winsorization quantiles for the manifest's outlier-flagged columns. Clipping
# touches only the extreme 0.1% tails; every clamp is logged (see _winsorize).
WINSOR_Q: tuple[float, float] = (0.001, 0.999)

# Dataset "kinds":
#   ticker_ts : date + ticker + value columns (per-name daily/monthly series)
#   wide_ts   : date + value columns, one row per date (market-level series)
#   long_ts   : date + a category column (etf / event) + values
#   snapshot  : a single as-of, one row per ticker
KIND_TICKER_TS = "ticker_ts"
KIND_WIDE_TS = "wide_ts"
KIND_LONG_TS = "long_ts"
KIND_SNAPSHOT = "snapshot"


@dataclass(frozen=True)
class DatasetSpec:
    """How to load one broad-pull dataset (byte facts are pinned in tests)."""

    name: str
    relpath: str
    kind: str
    date_col: str
    ticker_col: str | None = None
    category_col: str | None = None  # long_ts grouping key (etf / event)
    winsor_cols: tuple[str, ...] = ()


# vol_term_rv numeric columns flagged "outliers flagged" in the manifest.
_VOL_TERM_RV_WINSOR = (
    "atm_iv_30d",
    "atm_iv_60d",
    "atm_iv_90d",
    "atm_iv_180d",
    "atm_iv_365d",
    "atm_iv_730d",
    "rv_10d",
    "rv_20d",
    "rv_30d",
    "rv_60d",
    "rv_90d",
    "rv_120d",
    "rv_180d",
    "rv_260d",
)

SPECS: dict[str, DatasetSpec] = {
    # ---- volatility / options surface ----
    "iv_surface": DatasetSpec(
        "iv_surface", "iv_surface/sp500_iv_surface.csv.gz", KIND_TICKER_TS, "date", "ticker"
    ),
    "vol_term_rv": DatasetSpec(
        "vol_term_rv",
        "per_name/vol_term_rv.csv.gz",
        KIND_TICKER_TS,
        "date",
        "ticker",
        winsor_cols=_VOL_TERM_RV_WINSOR,
    ),
    # ---- event-gate inputs ----
    "macro_calendar": DatasetSpec(
        "macro_calendar",
        "macro_calendar/sp500_macro_calendar.csv",
        KIND_LONG_TS,
        "release_date",
        category_col="event",
    ),
    "macro_releases": DatasetSpec(
        "macro_releases",
        "macro_calendar/sp500_macro_releases.csv",
        KIND_LONG_TS,
        "date",
        category_col="event",
    ),
    # ---- vol / rates / cross-asset (market-level wide series) ----
    "vol_indices": DatasetSpec(
        "vol_indices", "macro_vol/sp500_vol_indices.csv", KIND_WIDE_TS, "date"
    ),
    "spx_correlation": DatasetSpec(
        "spx_correlation", "macro_vol/spx_correlation.csv", KIND_WIDE_TS, "date"
    ),
    "credit_spreads": DatasetSpec(
        "credit_spreads", "macro_vol/credit_spreads.csv", KIND_WIDE_TS, "date"
    ),
    "vix_futures_curve": DatasetSpec(
        "vix_futures_curve", "macro_vol/vix_futures_curve.csv", KIND_WIDE_TS, "date"
    ),
    "ois_sofr_curve": DatasetSpec(
        "ois_sofr_curve", "macro_rates/ois_sofr_curve.csv", KIND_WIDE_TS, "date"
    ),
    "real_yields": DatasetSpec("real_yields", "macro_rates/real_yields.csv", KIND_WIDE_TS, "date"),
    "fed_funds": DatasetSpec("fed_funds", "macro_rates/fed_funds.csv", KIND_WIDE_TS, "date"),
    "macro_surprise": DatasetSpec(
        "macro_surprise", "macro_rates/macro_surprise.csv", KIND_WIDE_TS, "date"
    ),
    "fx": DatasetSpec("fx", "macro_rates/fx.csv", KIND_WIDE_TS, "date"),
    "commodities": DatasetSpec("commodities", "macro_rates/commodities.csv", KIND_WIDE_TS, "date"),
    "global_vol": DatasetSpec("global_vol", "macro_rates/global_vol.csv", KIND_WIDE_TS, "date"),
    "sector_factor_etfs": DatasetSpec(
        "sector_factor_etfs",
        "macro_rates/sector_factor_etfs_ohlcv.csv",
        KIND_LONG_TS,
        "date",
        category_col="etf",
    ),
    # ---- per-name panels ----
    "returns_micro": DatasetSpec(
        "returns_micro", "per_name/returns_micro.csv", KIND_TICKER_TS, "date", "ticker"
    ),
    "beta_shares": DatasetSpec(
        "beta_shares",
        "per_name/beta_shares.csv",
        KIND_TICKER_TS,
        "date",
        "ticker",
        winsor_cols=("beta_raw",),
    ),
    "fundamentals_q": DatasetSpec(
        "fundamentals_q", "per_name/fundamentals_q.csv", KIND_TICKER_TS, "date", "ticker"
    ),
    "fundamentals_ext_q": DatasetSpec(
        "fundamentals_ext_q", "per_name/fundamentals_ext_q.csv", KIND_TICKER_TS, "date", "ticker"
    ),
    "estimates_m": DatasetSpec(
        "estimates_m", "per_name/estimates_m.csv", KIND_TICKER_TS, "date", "ticker"
    ),
    "estimates_fwd": DatasetSpec(
        "estimates_fwd", "per_name/estimates_fwd.csv", KIND_TICKER_TS, "date", "ticker"
    ),
    "valuation_m": DatasetSpec(
        "valuation_m",
        "per_name/valuation_m.csv",
        KIND_TICKER_TS,
        "date",
        "ticker",
        winsor_cols=("px_to_book", "ev_to_ebitda", "px_to_sales", "pe", "peg"),
    ),
    "options_sentiment": DatasetSpec(
        "options_sentiment",
        "per_name/options_sentiment.csv",
        KIND_TICKER_TS,
        "date",
        "ticker",
        winsor_cols=("pc_oi_ratio", "pc_vol_ratio", "news_sent"),
    ),
    "dividend_pit": DatasetSpec(
        "dividend_pit",
        "dividend_pit/sp500_dividend_yield_pit.csv",
        KIND_TICKER_TS,
        "date",
        "ticker",
    ),
    "short_interest": DatasetSpec(
        "short_interest",
        "short_interest/sp500_short_interest.csv",
        KIND_TICKER_TS,
        "date",
        "ticker",
    ),
    "snapshot_bdp": DatasetSpec(
        "snapshot_bdp", "per_name/sp500_snapshot_bdp.csv", KIND_SNAPSHOT, "asof", "ticker"
    ),
}


class BroadPullLoader:
    """Loader for the integrated broad-pull datasets (dormant; nothing consumes it)."""

    def __init__(self, data_dir: str | Path = BROAD_PULL_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._panels: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _read(self, relpath: str) -> pd.DataFrame | None:
        """Read a CSV/gz under ``data_dir``; ``None`` (+ warning) if absent.

        Pandas infers gzip from the ``.gz`` extension, so the same call handles
        ``iv_surface``/``vol_term_rv`` and the plain CSVs alike.
        """
        path = self.data_dir / relpath
        if not path.exists():
            logger.warning("broad-pull file not found: %s", path)
            return None
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    @staticmethod
    def _winsorize(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
        """Clip ``cols`` to their WINSOR_Q quantiles, logging every clamp.

        Never silent: when any value is clipped a WARNING records the column,
        the count, and the bounds. Row count is unchanged (clipping, not
        dropping), so downstream counts still match the manifest.
        """
        lo_q, hi_q = WINSOR_Q
        for col in cols:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            lo = s.quantile(lo_q)
            hi = s.quantile(hi_q)
            if pd.isna(lo) or pd.isna(hi):
                df[col] = s
                continue
            n_clamped = int(((s < lo) | (s > hi)).sum())
            if n_clamped:
                logger.warning(
                    "winsorized %d value(s) in %s to [%.6g, %.6g]", n_clamped, col, lo, hi
                )
            df[col] = s.clip(lo, hi)
        return df

    @staticmethod
    def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float64 columns to float32 to keep large panels memory-light."""
        float_cols = df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            df[col] = df[col].astype("float32")
        return df

    @staticmethod
    def _ticker_mask(df: pd.DataFrame, ticker_col: str, ticker: str) -> pd.Series:
        """Boolean mask selecting ``ticker`` by normalized symbol.

        Uses the precomputed ``ticker_normalized`` column when present (fast,
        for loaded panels); falls back to on-the-fly normalization of the raw
        column (for hand-built panels in unit tests).
        """
        key = normalize_ticker(ticker)
        if "ticker_normalized" in df.columns:
            return df["ticker_normalized"] == key
        return df[ticker_col].map(normalize_ticker) == key

    def _process(self, spec: DatasetSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the date column, normalize tickers, winsorize, downcast, sort."""
        if spec.date_col in df.columns:
            df[spec.date_col] = pd.to_datetime(df[spec.date_col], errors="coerce")
        # Bloomberg-style tickers ("AAPL UW") -> plain symbol ("AAPL"), so the
        # per-ticker accessors match the canonical universe (same normalization
        # as data/consolidated_loader). Added as a new column; the raw ticker
        # column is preserved (its distinct count is byte-pinned in tests).
        if (
            spec.ticker_col
            and spec.ticker_col in df.columns
            and spec.kind
            in (
                KIND_TICKER_TS,
                KIND_SNAPSHOT,
            )
        ):
            df["ticker_normalized"] = df[spec.ticker_col].map(normalize_ticker)
        if spec.winsor_cols:
            df = self._winsorize(df, spec.winsor_cols)
        df = self._downcast_floats(df)
        # Deterministic ordering for downstream PIT slicing.
        sort_keys = [k for k in (spec.ticker_col, spec.category_col, spec.date_col) if k]
        sort_keys = [k for k in sort_keys if k in df.columns]
        if sort_keys:
            df = df.sort_values(sort_keys).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # generic load / access
    # ------------------------------------------------------------------ #
    def load(self, name: str) -> pd.DataFrame | None:
        """Load (and cache) one dataset's full panel. ``None`` if file absent."""
        if name not in SPECS:
            raise KeyError(f"unknown broad-pull dataset: {name!r}")
        spec = SPECS[name]
        df = self._read(spec.relpath)
        if df is None:
            return None
        df = self._process(spec, df)
        self._panels[name] = df
        logger.info("loaded broad-pull %s: %d rows from %s", name, len(df), spec.relpath)
        return df

    def load_all(self) -> dict[str, int]:
        """Load every dataset; return a name→row-count map.

        Self-contained convenience for tests and the future supervised wiring —
        **not** wired into ``ConsolidatedBloombergLoader.load_all`` or any
        production path.
        """
        counts: dict[str, int] = {}
        for name in SPECS:
            df = self.load(name)
            counts[name] = 0 if df is None else len(df)
        return counts

    def panel(self, name: str) -> pd.DataFrame | None:
        """Return the full panel for ``name`` (loading it on first access)."""
        if name not in self._panels:
            return self.load(name)
        return self._panels[name]

    def series(
        self, name: str, ticker: str, as_of: str | pd.Timestamp | None = None
    ) -> pd.DataFrame | None:
        """Per-ticker time series for a ``ticker_ts`` dataset, optionally PIT-filtered.

        Lazy: filters the cached panel rather than holding per-ticker copies.
        ``as_of`` keeps only rows on/before that date (no-look-ahead helper for
        the future wiring; this loader itself feeds no consumer).
        """
        spec = SPECS[name]
        if spec.kind != KIND_TICKER_TS or spec.ticker_col is None:
            raise ValueError(f"series() is for ticker_ts datasets, not {name!r} ({spec.kind})")
        df = self.panel(name)
        if df is None:
            return None
        out = df[self._ticker_mask(df, spec.ticker_col, ticker)]
        if as_of is not None and spec.date_col in out.columns:
            out = out[out[spec.date_col] <= pd.to_datetime(as_of)]
        return out.reset_index(drop=True)

    def category_series(
        self, name: str, category: str, as_of: str | pd.Timestamp | None = None
    ) -> pd.DataFrame | None:
        """Rows for one category (etf / event) of a ``long_ts`` dataset."""
        spec = SPECS[name]
        if spec.kind != KIND_LONG_TS or spec.category_col is None:
            raise ValueError(f"category_series() is for long_ts datasets, not {name!r}")
        df = self.panel(name)
        if df is None:
            return None
        out = df[df[spec.category_col].astype(str) == category]
        if as_of is not None and spec.date_col in out.columns:
            out = out[out[spec.date_col] <= pd.to_datetime(as_of)]
        return out.reset_index(drop=True)

    def snapshot_row(self, name: str, ticker: str) -> dict | None:
        """Single as-of row for a ``snapshot`` dataset (e.g. ``snapshot_bdp``)."""
        spec = SPECS[name]
        if spec.kind != KIND_SNAPSHOT or spec.ticker_col is None:
            raise ValueError(f"snapshot_row() is for snapshot datasets, not {name!r}")
        df = self.panel(name)
        if df is None:
            return None
        hit = df[self._ticker_mask(df, spec.ticker_col, ticker)]
        return None if hit.empty else hit.iloc[0].to_dict()

    # ------------------------------------------------------------------ #
    # named load_* accessors (roadmap §10 idiom) — one per net-new file
    # ------------------------------------------------------------------ #
    def load_iv_surface(self) -> pd.DataFrame | None:
        return self.load("iv_surface")

    def load_vol_term_rv(self) -> pd.DataFrame | None:
        return self.load("vol_term_rv")

    def load_macro_calendar(self) -> pd.DataFrame | None:
        return self.load("macro_calendar")

    def load_macro_releases(self) -> pd.DataFrame | None:
        return self.load("macro_releases")

    def load_vol_indices(self) -> pd.DataFrame | None:
        return self.load("vol_indices")

    def load_spx_correlation(self) -> pd.DataFrame | None:
        return self.load("spx_correlation")

    def load_credit_spreads(self) -> pd.DataFrame | None:
        return self.load("credit_spreads")

    def load_vix_futures_curve(self) -> pd.DataFrame | None:
        return self.load("vix_futures_curve")

    def load_ois_sofr_curve(self) -> pd.DataFrame | None:
        return self.load("ois_sofr_curve")

    def load_real_yields(self) -> pd.DataFrame | None:
        return self.load("real_yields")

    def load_fed_funds(self) -> pd.DataFrame | None:
        return self.load("fed_funds")

    def load_macro_surprise(self) -> pd.DataFrame | None:
        return self.load("macro_surprise")

    def load_fx(self) -> pd.DataFrame | None:
        return self.load("fx")

    def load_commodities(self) -> pd.DataFrame | None:
        return self.load("commodities")

    def load_global_vol(self) -> pd.DataFrame | None:
        return self.load("global_vol")

    def load_sector_factor_etfs(self) -> pd.DataFrame | None:
        return self.load("sector_factor_etfs")

    def load_returns_micro(self) -> pd.DataFrame | None:
        return self.load("returns_micro")

    def load_beta_shares(self) -> pd.DataFrame | None:
        return self.load("beta_shares")

    def load_fundamentals_q(self) -> pd.DataFrame | None:
        return self.load("fundamentals_q")

    def load_fundamentals_ext_q(self) -> pd.DataFrame | None:
        return self.load("fundamentals_ext_q")

    def load_estimates_m(self) -> pd.DataFrame | None:
        return self.load("estimates_m")

    def load_estimates_fwd(self) -> pd.DataFrame | None:
        return self.load("estimates_fwd")

    def load_valuation_m(self) -> pd.DataFrame | None:
        return self.load("valuation_m")

    def load_options_sentiment(self) -> pd.DataFrame | None:
        return self.load("options_sentiment")

    def load_dividend_pit(self) -> pd.DataFrame | None:
        return self.load("dividend_pit")

    def load_short_interest(self) -> pd.DataFrame | None:
        return self.load("short_interest")

    def load_snapshot_bdp(self) -> pd.DataFrame | None:
        return self.load("snapshot_bdp")


_loader: BroadPullLoader | None = None


def get_broad_pull_loader() -> BroadPullLoader:
    """Return a process-wide :class:`BroadPullLoader` (lazy; loads nothing)."""
    global _loader
    if _loader is None:
        _loader = BroadPullLoader()
    return _loader
