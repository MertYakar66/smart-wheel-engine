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
| Conditional: before the next Flex trades update | The IBKR Flex token is IP-locked and rotates. If a pull answers `1012 Token invalid/expired`, the Operator regenerates it. | Operator | **OPEN**: last known good state unrecorded. On 2026-09-26 the Operator undertook to rotate it, because card 1 found an old copy of `flex_credentials.json` on Drive, which the Operator deletes by hand | `docs/DASHBOARD_TRADES.md` §1; `docs/DASHBOARD_TERMINAL.md` §3.3; `PROJECT_STATE.md` §0 B |

## Data dates

The pen's mark shows the oldest recorded frontier and its age. This table holds
the dates the manifest does not record, each with its source. Refresh it at
close.

| Dataset | Last date | Where it lives | Source |
| ------- | --------- | -------------- | ------ |
| Prices and IV (Bloomberg CSVs) | 2026-07-02, frozen: the Terminal is gone (D29) | the desktop root, `data/bloomberg/` | `data/DATA_MANIFEST.json` frontier |
| Earnings-calendar overlay | 2026-07-03 snapshot | the desktop root | `PROJECT_STATE.md` §0 C |
| IBKR portfolio snapshot | 2026-07-18 (`portfolio_snapshot.json`); the morning pull was retired on 2026-09-25 (card 1), and its two recorded runs, 2026-09-23 and 2026-09-25, had failed | the desktop root, `data_processed/ibkr/` | the round-3 and card-1 worklogs |
| Theta (the desktop: 17,188 of ~132,862 files) | unknown; the subscription has lapsed, and Theta is collected again later (D33) | the desktop root, `data_processed/theta/`; Drive `swe-local-only/theta`; and a far fuller upload in Drive `SmartWheelData/data_processed/theta`, found by card 1, whose size card 1b confirms. The Operator ruled on 2026-09-26 that Theta gets Bloomberg's rule: nothing that holds it is deleted until both copies are proven | `DECISIONS.md` D33; `PROJECT_STATE.md` §0 B; `docs/DATA_INVENTORY.md` §C.1, §C.3 |

*Last updated: 2026-09-26 (the Operator's rulings on card 1's findings: the Flex token row and the Theta row. Earlier, the close after #538: the IBKR snapshot row records the retired morning pull; the Theta row records the fuller Drive copy).*
