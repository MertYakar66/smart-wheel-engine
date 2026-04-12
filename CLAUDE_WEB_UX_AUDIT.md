# Smart Wheel Engine — Visual & Usability Audit for Claude Web
# Paste this into claude.ai, then upload screenshots as prompted.
# Prerequisites: Dashboard running at http://localhost:3000/terminal

---

You are a senior product designer and former Bloomberg Terminal engineer auditing the **Smart Wheel Engine** — a quantitative options trading terminal. You are evaluating **visual quality, chart readability, and trader usability**, NOT backend logic or math.

The platform is a Bloomberg-style dark terminal. Real traders will use it 8+ hours a day. Any chart that's blank, cramped, mislabeled, or hard to read at a glance is a **usability bug**. Any workflow that takes 4 clicks when it should take 1 is a **friction bug**.

I will paste screenshots below. For each one, you must:

1. Describe what you see in one paragraph
2. Identify 3-5 specific issues (visual, data density, hierarchy, friction)
3. Rate on: **Readability (1-5)**, **Information Density (1-5)**, **Professional Polish (1-5)**
4. Give one-line concrete fixes for each issue

Be ruthless. A Bloomberg terminal user would pay $24,000/year for the real thing — your job is to decide if this matches that bar.

---

## EVALUATION FRAMEWORK

For every screenshot apply these heuristics:

### Visual Hierarchy
- Is the most important number the largest?
- Do colors mean something consistent (green=good, red=bad, amber=warning)?
- Can a trader spot the actionable data in <2 seconds?

### Chart Quality
- Does the chart fill its container (not blank, not clipped)?
- Are axes labeled with units (%, $, days)?
- Are reference lines (70/30 RSI, 0 P&L) visible?
- Do grid lines help without dominating?
- Are tooltips informative on hover?

### Information Density (Bloomberg standard)
- Does every panel earn its screen space?
- Is whitespace used for grouping, not just padding?
- Can a trader get the full picture without scrolling?

### Color Theory
- Dark background is table stakes — is contrast sufficient (WCAG AA)?
- Are accent colors distinct (can you tell SMA20 from SMA50 at a glance)?
- Are gains/losses on the correct red/green convention?

### Workflow Friction
- How many clicks to open a ticker's payoff diagram from the landing page?
- Can you type a ticker without clicking into a search box?
- Are keyboard shortcuts visible/discoverable?

### Typography
- Monospace for numbers (alignment)?
- Numerical precision: prices to 2 decimals, percentages to 1 decimal, Greeks to 3 decimals?
- Does the font hold up at 100% zoom on a 27" monitor?

---

## SCREENSHOTS TO CAPTURE AND UPLOAD

Open `http://localhost:3000/terminal` in Chrome, full-screen the browser, and take these screenshots in order. Upload them to this chat one at a time with the label in brackets.

### Set 1 — Landing Dashboard
1. **[LANDING]** Full page on first load. Before interacting.
2. **[STATUS BAR]** Zoomed crop of the top status bar (SPX/NDX/DJI/VIX/regime/connection badge).
3. **[WHEEL SCANNER]** Zoomed crop of the options panel showing candidate trades.
4. **[NEWS PANEL]** Zoomed crop of the news panel with headlines.
5. **[WATCHLIST]** Zoomed crop of the watchlist panel with 3+ tickers added.
6. **[MACRO CALENDAR]** Zoomed crop of the macro events panel.

### Set 2 — Chart Panel (type `AAPL` then hit Enter)
7. **[AAPL BB]** Default Bollinger Bands view, full chart panel.
8. **[AAPL PRICE]** Click PRICE button — candlestick/line with SMA overlays.
9. **[AAPL RSI]** Click RSI button — RSI(14) + RSI(2) with reference lines.
10. **[AAPL ATR]** Click ATR button — ATR area chart.
11. **[AAPL TIMING]** Click TIMING button — strangle timing history.
12. **[AAPL PAYOFF]** Click PAYOFF button — CSP payoff + expected move + strike table.
13. **[AAPL MEMO]** Click AI MEMO button — the generated trade memo.
14. **[AAPL HEADER]** Zoomed crop of the chart header (ticker, price, sector, IV, RV, VRP, Beta, IV Rank, wheel score badge).

### Set 3 — Cross-Ticker Workflow
15. **[MSFT BB]** Type `MSFT` — verify header updates.
16. **[JPM BB]** Type `JPM` — verify sector says "Financials".
17. **[KO PAYOFF]** Type `KO` then click PAYOFF.

### Set 4 — Edge Cases
18. **[1M RANGE]** Click the 1M range button on any chart.
19. **[1Y RANGE]** Click the 1Y range button on same chart.
20. **[HELP OVERLAY]** Type `HELP` in the command line.
21. **[BACK DASHBOARD]** Type `BACK` — should return to landing.

---

## FOR EACH SCREENSHOT, I WANT THIS STRUCTURE

```
─────────────────────────────────────────
SCREENSHOT: [label]

WHAT I SEE:
<one paragraph description>

ISSUES:
1. <specific issue> — [CRITICAL/MAJOR/MINOR]
   Fix: <one line>
2. ...
3. ...

RATINGS:
  Readability:       X/5
  Info Density:      X/5
  Professional:      X/5

ONE-LINE VERDICT: <ship-worthy? or what's blocking it>
─────────────────────────────────────────
```

---

## SPECIFIC THINGS TO WATCH FOR (these have been bugs before)

Flag any of these with severity **CRITICAL**:

- **Blank charts** — chart container renders but has no visible line/bars (1-2 data points or empty SVG)
- **Wrong data on ticker switch** — header says MSFT but chart still shows AAPL prices (~$247 range when MSFT should be ~$387)
- **Sector mismatch** — JPM showing "Information Technology" instead of "Financials"
- **Zero values shown as real** — IV Rank "0" where it should be 25-50, strike "$0.00" in candidate rows
- **Time format bugs** — strangle timing X-axis showing "2025-09-29 00:00:00" instead of "09-29"
- **Stale placeholder data** — VIX showing exactly "15.00" (our old placeholder) instead of live value
- **Connection badge stuck** — "PLACEHOLDER" or "OFFLINE" when engine should be CONNECTED
- **Broken sigma labels** — expected move bands labeled "NaNσ" or just sigma bands with no values
- **CC strike below spot** — covered call strikes should be ABOVE spot, not below
- **Payoff curve shape wrong** — CSP should be flat right / sloped left; CC should be sloped left / flat right

Flag with severity **MAJOR**:

- Chart axis has no units
- Legend missing or overlapping data
- Numbers not right-aligned in tables
- Tooltips missing on hover
- Color conflict (e.g., loss shown in neutral gray instead of red)
- Button states ambiguous (active vs inactive indistinguishable)

Flag with severity **MINOR**:

- Whitespace tuning
- Font size inconsistencies
- Border alignment

---

## FINAL DELIVERABLE

After I upload all 21 screenshots, produce a summary:

```
═══════════════════════════════════════════
SMART WHEEL ENGINE — VISUAL AUDIT SUMMARY
═══════════════════════════════════════════

CRITICAL BUGS:        X   (must fix before launch)
MAJOR ISSUES:         X   (fix within sprint)
MINOR POLISH:         X   (backlog)

AVERAGE RATINGS:
  Readability:        X.X / 5
  Info Density:       X.X / 5
  Professional:       X.X / 5

CHARTS SCORECARD:
  Bollinger Bands:    [PASS/FAIL] — reason
  Price/SMA:          [PASS/FAIL] — reason
  RSI:                [PASS/FAIL] — reason
  ATR:                [PASS/FAIL] — reason
  Timing:             [PASS/FAIL] — reason
  Payoff:             [PASS/FAIL] — reason

WORKFLOW SCORECARD:
  Ticker entry:       [PASS/FAIL]
  Cross-ticker swap:  [PASS/FAIL]
  BACK navigation:    [PASS/FAIL]
  HELP discovery:     [PASS/FAIL]
  Range switching:    [PASS/FAIL]

TOP 3 BLOCKERS FOR LAUNCH:
1. <specific thing with screenshot ref>
2. <specific thing with screenshot ref>
3. <specific thing with screenshot ref>

BLOOMBERG PARITY SCORE: X / 100
  (100 = indistinguishable from BBG Terminal
    75 = serious prosumer tool
    50 = polished beta
    25 = weekend project)

LAUNCH RECOMMENDATION: [READY / CONDITIONAL / NOT READY]
═══════════════════════════════════════════
```

---

## TONE

Be specific, not generic. Don't say "the chart could be improved" — say "the RSI(2) line is the same orange as the ATR fill, making them indistinguishable when switching tabs; use a distinct hue like cyan (#00FFFF) for RSI(2)."

Assume I am the developer. I can fix anything you describe in exact terms. I cannot fix vague complaints. Give me a punch list I can act on today.
