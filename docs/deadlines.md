# Deadlines and data dates — what is due, and how old the data is

**Read at session-open (`CLAUDE.md` §2).** The nearest open row of the table
goes in the pen's first line. The age of the data comes from
`data/DATA_MANIFEST.json`, not from here. Update this file at session-close:
add what is new, move what moved, and close what closed. A row closes on
evidence, never on a verbal assurance.

Rules for this file:
- Only a date that a source states. No estimates.
- Every row cites the document that establishes it.
- A row closes when the repository or a pasted record proves it, not before.
- Credentials (tokens, logins, account numbers) are never written here.

| Due | What | Owner | Status | Source |
| --- | ---- | ----- | ------ | ------ |
| Undated | ThetaData subscription: when does it renew or lapse? The laptop's Theta corpus and every Theta pull depend on it. | Operator: supply the date | **CLOSED 2026-09-24**: the subscription is no longer active (the Operator, recorded in D33). Theta is collected again later, from a source not yet chosen | `docs/LAPTOP_SETUP.md` §3 (the tier record); `DECISIONS.md` D33 |
| Conditional: before the next Flex trades update | The IBKR Flex token is IP-locked and rotates. If a pull answers `1012 Token invalid/expired`, the Operator regenerates it. | Operator | **OPEN**: last known good state unrecorded | `docs/DASHBOARD_TRADES.md` §1; `docs/DASHBOARD_TERMINAL.md` §3.3 |

## Data dates

The pen's mark shows the oldest recorded frontier and its age. This table holds
the dates the manifest does not record, each with its source. Refresh it at
close.

| Dataset | Last date | Where it lives | Source |
| ------- | --------- | -------------- | ------ |
| Prices and IV (Bloomberg CSVs) | 2026-07-02, frozen: the Terminal is gone (D29) | the desktop root, `data/bloomberg/` | `data/DATA_MANIFEST.json` frontier |
| Earnings-calendar overlay | 2026-07-03 snapshot | the desktop root | `PROJECT_STATE.md` §0 C |
| IBKR portfolio snapshot | 2026-07-18 (`portfolio_snapshot.json`); the morning pull that refreshed it is being retired | the desktop root, `data_processed/ibkr/` | the desktop's round-3 worklog, 2026-09-23 |
| Theta (partial: 17,188 of ~132,862 files) | unknown; the subscription has lapsed, and Theta is collected again later (D33) | the desktop root, `data_processed/theta/`; also Drive `swe-local-only/theta` | `DECISIONS.md` D33; `docs/DATA_INVENTORY.md` §C.1 |

*Last updated: 2026-09-24.*
