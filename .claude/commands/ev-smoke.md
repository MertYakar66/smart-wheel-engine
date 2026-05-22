---
description: Run the 5-ticker EV-ranker smoke check from CLAUDE.md §6
---

Run the data-layer smoke check from `CLAUDE.md` §6 — it confirms the
Bloomberg-CSV + connector + EV-engine path is healthy without the
slow full-universe `scripts/diagnose_candidates.py` run.

Execute exactly:

```python
from engine.wheel_runner import WheelRunner

runner = WheelRunner()
print("connector:", type(runner.connector).__name__)

df = runner.rank_candidates_by_ev(
    tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
    top_n=10,
    min_ev_dollars=-1e9,
    include_diagnostic_fields=True,
)
print(df[["ticker", "ev_dollars", "iv", "premium"]])
```

The path is healthy if it returns 5 rows with non-null `ev_dollars`,
`iv`, and `premium`. Report the result, and report which connector class
was selected (`MarketDataConnector` for the `bloomberg` provider,
`ThetaConnector` for `theta`) — silent provider selection is a known bug
source per `CLAUDE.md` §3.

This command is a thin wrapper around the documented §6 check; it adds no
logic of its own.
