"""Split-adjustment layer for joining engine candidates to the Theta larder.

The two price spaces do **not** agree:

* The engine's spot/strike come from Bloomberg OHLCV, which is **split
  *adjusted* to the most recent split** (e.g. AAPL trades at ~$75 on
  2020-01-08 in adjusted space).
* The Theta larder stores **raw, as-listed contract strikes** (e.g. a
  $300 AAPL put on 2020-01-08, before the Aug-2020 4:1 split).

Joining an engine-solved strike (adjusted) straight onto a larder strike
(raw) silently mis-matches whenever a split sits between ``as_of`` and
today. The pilot's first proof-of-pipeline point did exactly this — matched
an adjusted-$74 strike to a raw-$75 strike on a stock then trading ~$300 raw
— and printed a bogus ``-99%`` "premium correction". This module exists to
make that join correct and to be unit-tested independently (the post-split
pilot band exercises the identity path only, so the split path must be
validated against a known split — see ``tests/test_premium_correction_pilot``).

Convention for an N:1 forward split with ex-date ``D``:

* A price/strike that is **adjusted** (post-all-splits) maps to its **raw**
  value on a date *before* ``D`` by multiplying by ``N``::

      raw = adjusted * cumulative_factor_after(as_of)

  where ``cumulative_factor_after`` is the product of every split ratio with
  ex-date strictly after ``as_of``.
* An option **premium** scales the same way as the underlying: a raw
  premium quoted pre-split is divided by the cumulative factor to express it
  in adjusted (per-adjusted-share) dollars::

      adjusted_premium = raw_premium / cumulative_factor_after(as_of)

In the post-split pilot band (2024-09 onward for these three names) the
cumulative factor is ``1.0`` and every conversion is the identity — which is
*why* the band was chosen, and why the layer is tested against 2020 data
rather than the band it ships against.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# Forward-split table for the pilot names (and the immediate full-study
# neighbours). Each entry is ``(ex_date, ratio)`` for an N:1 forward split.
# Sources: public corporate-action records. Ratios are the N in "N:1".
#
# Keep this table explicit and auditable rather than deriving factors from a
# Bloomberg adjusted/unadjusted ratio — for a pilot, a transparent hand-
# checked table is safer than an inferred one (and the larder carries no
# underlying spot to infer from at this tier).
_SPLITS: dict[str, list[tuple[date, float]]] = {
    "AAPL": [(date(2020, 8, 31), 4.0)],
    "TSLA": [(date(2020, 8, 31), 5.0), (date(2022, 8, 25), 3.0)],
    "NVDA": [(date(2021, 7, 20), 4.0), (date(2024, 6, 10), 10.0)],
    # --- full-study neighbours (not used by the 3-name pilot) ---
    "AMZN": [(date(2022, 6, 6), 20.0)],
    "GOOGL": [(date(2022, 7, 18), 20.0)],
    "GOOG": [(date(2022, 7, 18), 20.0)],
}


def _as_date(d: date | str | pd.Timestamp) -> date:
    if isinstance(d, date) and not isinstance(d, pd.Timestamp):
        return d
    return pd.Timestamp(d).date()


def cumulative_factor_after(ticker: str, as_of: date | str | pd.Timestamp) -> float:
    """Product of every split ratio with ex-date strictly after ``as_of``.

    Returns ``1.0`` when the ticker has no recorded splits, or none after
    ``as_of`` (the post-split identity path).
    """
    aod = _as_date(as_of)
    factor = 1.0
    for ex_date, ratio in _SPLITS.get(ticker.upper(), []):
        if ex_date > aod:
            factor *= ratio
    return factor


def adjusted_to_raw_strike(
    ticker: str, as_of: date | str | pd.Timestamp, adjusted_strike: float
) -> float:
    """Map an engine (adjusted) strike into the larder's raw strike space."""
    return adjusted_strike * cumulative_factor_after(ticker, as_of)


def raw_to_adjusted_strike(
    ticker: str, as_of: date | str | pd.Timestamp, raw_strike: float
) -> float:
    """Map a larder (raw) strike back into the engine's adjusted space."""
    return raw_strike / cumulative_factor_after(ticker, as_of)


def raw_to_adjusted_price(ticker: str, as_of: date | str | pd.Timestamp, raw_price: float) -> float:
    """Express a raw (as-listed) option premium in adjusted dollars.

    Premiums scale with the underlying: a pre-split premium is divided by the
    cumulative factor to compare against the engine's adjusted ``BSM(iv)``.
    """
    return raw_price / cumulative_factor_after(ticker, as_of)
