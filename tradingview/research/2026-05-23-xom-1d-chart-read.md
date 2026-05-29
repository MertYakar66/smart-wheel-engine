# XOM 1D — chart read

**Symbol:** NYSE:XOM (Exxon Mobil Corporation) — quoted via BATS:XOM on TradingView
**As of:** 2026-05-22 close (last bar; today 2026-05-23 is Saturday)
**Source:** TradingView Desktop (MCP, CDP attach)
**Screenshot:** `tradingview/tradingview-mcp-jackson/screenshots/XOM-1D-2026-05-23.png`

> **Format note:** `tradingview/CLAUDE.md` calls for `.docx`. Saved as `.md` because this session has no docx writer installed — convert with `pandoc 2026-05-23-xom-1d-chart-read.md -o 2026-05-23-xom-1d-chart-read.docx` when needed.

## Snapshot

| Field | Value |
|---|---|
| Last close | **154.92** |
| Day OHLC | O 154.03 / H 155.55 / L 153.17 / C 154.92 (range $2.38, the narrowest of the last 5 sessions) |
| Day volume | 12.95M — ~38% below the 100-bar avg of 20.96M |
| Bollinger Bands (20, 2) | Upper 161.75 / Basis 153.05 / Lower 144.35 |
| BB width | $17.40 (≈11.4% of basis) — moderate vol regime |
| Position in band | ~61% from lower, sitting just above the 20-SMA basis |

## 100-bar context (Dec 30 2025 → May 22 2026, ~5 months)

- Range: $118.27 → $176.41 ($58.14 swing, +27.9% top-to-bottom from open of the window).
- Trend shape on the chart: parabolic run from ~$118 to a high near $176 mid-window, then a sharp pullback that retested the basis, then a second leg up that peaked around $163 in early May, and now a pullback that has carried price back to the BB basis at $153.
- Last 5 sessions: 160.49 → 162.55 → 156.28 → 155.29 → 154.92. Three consecutive down closes, each with narrower true range — the kind of compression that resolves either as a low-volume drift higher or a basis break.

## Read

- **Tactical:** Price is mean-reverting toward the 20-SMA basis with declining volume. That's typically consolidation rather than capitulation — but a clean close below ~$153 (basis) opens the door to a retest of the lower band at ~$144.
- **Structural:** The two earnings markers (E icons) visible on the chart suggest a recent print and a forward one inside the window — chart-only data can't tell us which leg attributed to. Confirm via earnings calendar before sizing any new exposure.
- **Wheel angle (not a trade decision):** Short-put strikes that line up with technical support are $150 (round, near basis) and $145 (just above the lower band). Both would need the SWE EV engine to clear them — TradingView gives us no IV, no skew, no event lockout, and no greeks. This note ends at "technically interesting levels."

## Gaps this read does not cover

- IV / IV rank / skew → needs the SWE option-chain pull.
- Earnings date confirmation → needs an earnings calendar source (TradingView icons are unreliable for dates).
- Dealer positioning (gamma walls, flip) → needs the SWE dealer overlay.
- Fundamentals (FCF, dividend coverage, buyback cadence) → out of TradingView scope.

## Hard contract reminder

This is a **chart sanity check**, not a recommendation. Per `CLAUDE.md` §2, only `EVEngine.evaluate` ranks tradeable candidates. Nothing in this note can rescue a negative-EV candidate or bypass the decision layer.
