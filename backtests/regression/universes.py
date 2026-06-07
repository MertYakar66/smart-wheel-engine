"""Fixed universes used by the four regression backtests.

``UNIVERSE_24`` is the S17/S22/S27/S32/S35 set: 25 names from S17's
seven-sector watchlist with ``WMT`` dropped (S22's correction; ``WMT``
OHLCV history only goes back to 2025-12 on the dev box). Hardcoded
because snapshots locked to this list must remain reproducible even
if the connector's universe changes.

``UNIVERSE_100`` is the first 100 tickers from
``MarketDataConnector().get_universe()[:100]`` snapshotted at
``origin/main`` commit ``8a17b0b``. ``test_universes_match_connector``
(landing in PR2) asserts the derivation still holds on the current
connector.

Both are ``tuple[str, ...]`` (immutable) and importable without
touching disk — the connector is not called at module-load time.
"""

UNIVERSE_24: tuple[str, ...] = (
    # IT (5)
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "ORCL",
    # Financials (4)
    "JPM",
    "BAC",
    "GS",
    "WFC",
    # Energy (2)
    "XOM",
    "CVX",
    # Health Care (4)
    "UNH",
    "JNJ",
    "PFE",
    "MRK",
    # Staples (3 — S17 had 4, WMT dropped in S22)
    "PG",
    "KO",
    "COST",
    # Industrials (4)
    "CAT",
    "BA",
    "GE",
    "HON",
    # Discretionary (2)
    "AMZN",
    "HD",
)

UNIVERSE_100: tuple[str, ...] = (
    "A",
    "AAPL",
    "ABBV",
    "ABNB",
    "ABT",
    "ACGL",
    "ACN",
    "ADBE",
    "ADI",
    "ADM",
    "ADP",
    "ADSK",
    "AEE",
    "AEP",
    "AES",
    "AFL",
    "AIG",
    "AIZ",
    "AJG",
    "AKAM",
    "ALB",
    "ALGN",
    "ALL",
    "ALLE",
    "AMAT",
    "AMCR",
    "AMD",
    "AME",
    "AMGN",
    "AMP",
    "AMT",
    "AMZN",
    "ANET",
    "AON",
    "AOS",
    "APA",
    "APD",
    "APH",
    "APO",
    "APP",
    "APTV",
    "ARE",
    "ARES",
    "ATO",
    "AVB",
    "AVGO",
    "AVY",
    "AWK",
    "AXON",
    "AXP",
    "AZO",
    "BA",
    "BAC",
    "BALL",
    "BAX",
    "BBY",
    "BDX",
    "BEN",
    "BF/B",
    "BG",
    "BIIB",
    "BK",
    "BKNG",
    "BKR",
    "BLDR",
    "BLK",
    "BMY",
    "BNY",
    "BR",
    "BRK/B",
    "BRO",
    "BSX",
    "BX",
    "BXP",
    "C",
    "CAG",
    "CAH",
    "CARR",
    "CASY",
    "CAT",
    "CB",
    "CBOE UF",
    "CBRE",
    "CCI",
    "CCL",
    "CDNS",
    "CDW",
    "CEG",
    "CF",
    "CFG",
    "CHD",
    "CHRW",
    "CHTR",
    "CI",
    "CIEN",
    "CINF",
    "CL",
    "CLX",
    "CMCSA",
    "CME",
)

assert len(UNIVERSE_24) == 24, f"UNIVERSE_24 must be 24 tickers, got {len(UNIVERSE_24)}"
assert len(UNIVERSE_100) == 100, f"UNIVERSE_100 must be 100 tickers, got {len(UNIVERSE_100)}"
