# Syllabus: from understanding to CFA-level mastery

**For:** the Operator. **Prepared by:** the Portfolio Manager / Risk Engineer. **Status:** approved 2026-10-15 (`DECISIONS.md` PM3). Before approval, two fresh agents checked the draft, and a skeptic confirmed 17 of their 20 findings. All 17 are fixed.

**Built on the prompt you approved.**
- **Backbone:** the CFA program.
- **Order:** what protects you while you trade comes first: risk measures and position sizing, then the Greeks and implied volatility, then portfolio maths, then the rest of the CFA.
- **Real portfolio work comes before study** ("lets prioritize the real portfolio work because it is real money").
- **Limit:** worked examples show a method on your numbers and never end in a recommended size or trade. Every limit and every trade is your decision.

**Where the facts come from** (checked 2026-10-15):
- cfainstitute.org: the exam pages, dates and fees, policies, and the 2027 topic outlines, read directly.
- IBKR: the Portfolio Margin page and the options-margin page for residents of Canada, read directly. Its knowledge-base articles were read through search results.
- Books: publisher, retailer and library catalogue pages.
- Cboe, AQR, BIS and the Federal Reserve: their own pages, with the Fed's figures read through search results.

---

## 1. Two tracks, side by side

| Track | What | Hours | When |
| --- | --- | --- | --- |
| **1. Desk skills** | The money-critical topics, taught on your real account. Some are **not named in the 2027 CFA topic outlines**: broker margin and liquidation, Kelly sizing, risk of ruin, and volatility targeting. The CFA teaches scenario risk measures at Level II, but not a stress routine for your own book. | about 92 h | Weeks 1–6 |
| **2. The CFA path** | The full Level I curriculum (2027 edition, for exams from February 2027), then Levels II and III. | about 325 h for Level I plus a 15 h Practical Skills Module (CFA Institute: successful candidates report "over 300 hours" per level) | Light from week 1, full pace from week 7 |

**The hours basis.**
- The calendar counts the time spent on lessons and reading or practice. Applying a lesson to your account, and the spaced review, come on top.
- Real portfolio work comes first, so the plan counts **about 3 h a day**, not your full 5–6 h. If you study more, the dates move earlier (§4).
- On a heavy desk day, study shrinks and the dates move. The plan doesn't break.

---

## 2. Track 1: desk skills (weeks 1–6)

**Your account cases wait for the read-only account read (the IBKR round).**
- Until then, each lesson uses the engine's data or a hypothetical example, and a module can close on that.
- Once the read exists, each module gets one budgeted re-run on your own account (about 1 h each).
- Account details and worked answers stay in `pm-private`.

### M1. Risk measures (target 2.5; you rated 0.5), about 24 h

| # | Lesson | The case |
| --- | --- | --- |
| 1.0 | **The toolkit** (2–3 h, intuition only): delta-dollars, vega as sensitivity to implied volatility, beta, correlation. M3–M5 teach them fully. | Engine examples |
| 1.1 | **Margin and liquidation at IBKR:** excess liquidity, the maintenance requirement, real-time liquidation without a margin call, why a deposit does not delay it, and liquidation from post-expiry exposure. IBKR states: "Customers in Canada are not eligible for Portfolio Margin accounts", and its options-margin page for residents of Canada trading US options says "Rules-based margin". The IBKR round confirms which entity holds your account and which of IBKR's pages apply. | A hypothetical assignment. A real one from your account is added once the account read confirms it |
| 1.2 | **Returns and volatility,** built only from a series adjusted for deposits and withdrawals (IBKR PortfolioAnalyst's time-weighted return, or account value adjusted for each cash flow). Then annualising, and the square root of time. | Engine data; then your account |
| 1.3 | **Drawdown and recovery:** why −50% needs +100%; maximum drawdown and its duration | Engine backtests; then your account |
| 1.4 | **Leverage and exposure:** gross and net, beta-adjusted; concentration (HHI, the effective number of positions) | Against the engine's 10% single-name and 25% sector caps |
| 1.5 | **Value at Risk (VaR):** parametric, historical and Monte Carlo, and what it does not tell you | Engine data; then your book |
| 1.6 | **Expected shortfall (CVaR):** fat tails, and why regulators moved to it (Basel uses 97.5%) | The engine's `cvar_5` and its known over-confidence |
| 1.7 | **Stress tests and scenarios:** a −10% / −20% / −58% market (the Fed's 2026 severely adverse scenario), a VIX spike, an assignment wave | The engine's R8 check, which ignores stock legs |

1.5 and 1.7 are re-run once after M5 (2 h), when the full tools are in hand.

### M2. Position sizing (target 2.5; you rated 1), about 14 h

| # | Lesson | The case |
| --- | --- | --- |
| 2.1 | Risk per trade; fixed-fractional sizing | A hypothetical cash-secured put |
| 2.2 | Volatility targeting, and the evidence for it (Harvey et al. 2018; Moreira–Muir 2017) | Sizing by a name's volatility |
| 2.3 | Kelly and fractional Kelly; why full Kelly is dangerous when your estimates are wrong | The engine's over-confident top bin as the estimation error |
| 2.4 | Risk of ruin; sizing short options by notional, not premium; assignment as a sizing event | A put's notional against its premium |
| 2.5 | Portfolio limits: the first draft of your investment policy statement (IPS) | Your limits, which you decide |

### M3. The Greeks (target 3; you rated 1.5), about 16 h

| # | Lesson | The case |
| --- | --- | --- |
| 3.1 | Delta: hedge ratio, rough probability, delta-dollars | The engine's 0.25-delta put |
| 3.2 | Gamma: why short options lose fast near expiry | 7-day against 28-day options (D29's menu) |
| 3.3 | Theta: the wheel's income, and the theta–gamma trade-off | Engine examples; then your book |
| 3.4 | Vega and rho | A book in an IV spike |
| 3.5 | Portfolio Greeks: adding up a book | Engine examples; then your book |

### M4. Implied volatility and the volatility risk premium (target 2.5; you rated 1), about 14 h

| # | Lesson | The case |
| --- | --- | --- |
| 4.1 | What implied volatility is; implied against realised | One S&P 500 name |
| 4.2 | The volatility risk premium: why selling options earns on average (Cboe PUT index; Bondarenko 2019; AQR 2018), and why the edge is lower volatility and drawdown, not higher return (Bondarenko 2016 and its critics) | The wheel against the S&P 500 |
| 4.3 | Skew and the surface: why puts cost more | The engine's Bloomberg IV, which has no skew |
| 4.4 | Events: IV around earnings, and IV crush | The engine's earnings lockout |

### M5. Portfolio maths (target 2.5; you rated 1), about 17 h

| # | Lesson | The case |
| --- | --- | --- |
| 5.1 | Arithmetic and log returns; time-weighted against money-weighted (TWR against MWR, GIPS 2020) | Engine data; then your account |
| 5.2 | Covariance, correlation, diversification; correlations rising in a crisis | Engine data; then your holdings |
| 5.3 | Sharpe, Sortino and Calmar; choosing a benchmark | Your account against a broad index and a sector benchmark |
| 5.4 | CAPM, beta, alpha and its t-statistic: skill or luck | A skill-vs-luck test |
| 5.5 | Factor models: the market plus a sector factor; the French data library | Engine data; then your book |

**Track 1 total: about 92 h:**
- the five modules, 85 h;
- the 1.5 and 1.7 re-run, 2 h;
- one account re-run per module, 5 h.

---

## 3. Track 2: the CFA path

**Level I, 2027 curriculum:** 180 multiple-choice questions in two sessions of 135 minutes. The ten-year average pass rate is 41%.

| Topic (2027 name) | Weight | Order | Hours (about) |
| --- | --- | --- | --- |
| Ethical and Professional Standards | 10–15% | Weekly from week 1 (Standards I–VII) | 35 |
| Quantitative Methods | 11–14% | First (overlaps M1 and M5) | 35 |
| Portfolio Construction | 8–12% | Second (overlaps M5) | 25 |
| Derivatives and Risk Management | 6–9% | Third (overlaps your options basics and M3) | 20 |
| Equities | 11–14% | | 35 |
| Fixed Income | 11–14% | | 35 |
| Financial Statement Analysis | 11–14% | | 40 |
| Corporate Finance | 6–9% | | 20 |
| Economics | 6–9% | | 25 |
| Alternative Investments | 6–9% | | 15 |
| Practical Skills Module (Python Programming Fundamentals recommended; Python is what the engine is built in) | required to receive your result | Any time after registration, no later than the results release; placed in the review gap | 15 |
| Mock exams and final review, including the two official mocks (released about 60 days before your window) | | The last 4–6 weeks before the exam | 40 |
| **Total** | | | **about 340** |

**Weeks 1–3, before registration.** The official curriculum comes only with registration. Until then:
- the free 2027 Level I topic outlines are the checklist;
- Bodie, Kane and Marcus, chapters 5–9, cover the Quant and portfolio overlap.

The official material takes over once you register.

**Level II** (88 questions in 22 vignette item sets; pass rate 45%) has the advanced versions of your desk topics: "Valuation of Contingent Claims" (the Greeks, delta hedging, IV) and "Measuring and Managing Market Risk" (VaR, scenario risk, position and stop-loss limits).

**Level III** (item sets plus typed essays; pass rate 52%) covers options strategies and performance measurement. You choose a pathway: Portfolio Management, Private Markets or Private Wealth.

---

## 4. Calendar and exam choice (your decision)

**Starting point:** week 1 begins Mon 19 Oct 2026. Before the mock phase come 392 h: Track 1 (92), CFA content (285) and the PSM (15). Then 40 h of mocks, in the last 4–6 weeks before the exam.

| Study hours counted a day | Days a week | Content ends | Feb 2027 (22–28 Feb) | May 2027 (11–17 May) | Aug 2027 (15–22 Aug) |
| --- | --- | --- | --- | --- | --- |
| 3 h | 5 | about 20 Apr 2027 | misses | misses (by about a week) | fits |
| 3 h | 7 | about 27 Feb 2027 | misses | fits | fits |
| 4.5 h | 5 | about 18 Feb 2027 | misses | fits | fits |
| 4.5 h | 7 | about 14 Jan 2027 | fits (just) | fits | fits |
| 5.5 h | 7 | about 29 Dec 2026 | fits | fits | fits |

| Window | Register by | Fee |
| --- | --- | --- |
| Feb 2027 | 5 Nov 2026 (21 days from today) | USD 1,490 |
| May 2027 | 10 Feb 2027 (the early price ended 14 Oct 2026) | USD 1,490 |
| Aug 2027 | early price by 20 Jan 2027 (registration opens 3 Nov 2026, closes 29 Apr 2027) | **USD 1,140** early |

**My recommendation: August 2027**, registered soon after 3 November 2026:
- it fits every hours scenario;
- it is USD 350 cheaper;
- it leaves the most room for real portfolio work;
- the official curriculum, its 1,000+ practice questions and the mocks come only with registration, so registering early gets them to you sooner.

February is possible only at about 4.5+ h a day, seven days a week, with registration by 5 November. Registering and paying is yours to do.

**For August, the months after the content ends:**
- spaced review of Level I;
- a head start on Level II's two desk topics (Valuation of Contingent Claims; Measuring and Managing Market Risk);
- the PSM;
- from mid-June, the mock phase, with both official mocks.

**Rules:** each level may be sat at most twice a calendar year, never in back-to-back windows, and never in windows six months or less apart. All three levels cost USD 3,520–4,570 in exam fees.

---

## 5. How progress is measured

**Rating scale** (the same rubric for your self-rating and mine):

| Rating | Means |
| --- | --- |
| 0 | Never heard of it |
| 1 | Can explain the idea in plain words |
| 2 | At least 70% on the module test |
| 2.5 | At least 80% on the module test, and computes it unaided on real data |
| 3 | At least 90%, and can teach it back to me |

**The checks:**
- **Each lesson** ends with one check question.
- **Each week** ends with a test of 10–15 questions, mixing new topics with old ones (spaced review).
- **Each module** ends with a test of 15–20 questions.
- **A module is done** when you reach its target rating on that test and can explain the topic back in your own words.

**CFA:** your accuracy on practice questions, topic by topic, and the official mocks. A common rule of thumb from prep providers, not CFA Institute, is about 70% or more on mocks before you sit.

**The progress file** records, for each lesson: the date, the lesson number, pass or fail, the test score as a percentage, your rating and mine. It holds nothing else. Worked answers for account cases go only in `pm-private\library`.

**Where the files live:**
- The syllabus goes in `methods/learning.md`, and the progress file in `methods/learning-progress.md`, on the `Risk_Management` branch. The branch's allow-list already accepts these paths, so no new root file and no `.gitignore` change is needed.
- Lesson notes and your materials go in `pm-private\library`.

---

## 6. Reading list

You don't need to buy anything to start. Free sources are marked "free".

- **M1, risk measures:**
  - IBKR's knowledge-base articles on liquidation, deposits, expiration and margin warnings, plus its options-margin page for residents of Canada (free);
  - Hull, *Risk Management and Financial Institutions*, 6th edition (2023), chapters 8, 11–13, 16, 21 and 30;
  - Bodie, Kane and Marcus, *Investments*, 13th edition, chapter 5, as a gentle start;
  - Acerbi and Tasche (2002) on expected shortfall (free);
  - the Fed's 2026 stress scenarios (free);
  - the OCC's options risk disclosure (free).
- **M2, position sizing:**
  - Kelly (1956) (free);
  - Thorp's Kelly chapter (the 1997 version is free);
  - MacLean, Thorp and Ziemba (2010) on fractional Kelly (free);
  - Sinclair, *Positional Option Trading*, chapter 9, "Trade Sizing";
  - Carver, *Systematic Trading*, the chapters on volatility targeting and sizing;
  - Harvey et al. (2018) on volatility targeting (free);
  - Poundstone, *Fortune's Formula*, for the story.
- **M3, the Greeks:**
  - Hull, *Options, Futures, and Other Derivatives*, 11th edition, chapters 10–13, 15 and 19;
  - the Options Industry Council's pages on each Greek (free);
  - Natenberg, *Option Volatility and Pricing*, 2nd edition.
- **M4, implied volatility and the volatility risk premium:**
  - Hull, *Options, Futures, and Other Derivatives*, chapter 20;
  - Sinclair, *Volatility Trading*, 2nd edition, chapters 2–7;
  - Bondarenko (2019, 2016), with the critique that its figures are before costs;
  - AQR, "Understanding the Volatility Risk Premium" (2018) (free);
  - Israelov and Nielsen, "Covered Calls Uncovered" (2015);
  - Cboe's PUT index and VIX methodologies (free).
- **M5, portfolio maths:**
  - Bodie, Kane and Marcus, chapters 5–9 and 24;
  - Sharpe, "The Sharpe Ratio" (1994) (free);
  - GIPS 2020 on time-weighted and money-weighted returns (free);
  - the Kenneth French Data Library (free).
- **Later:**
  - Gatheral, *The Volatility Surface*;
  - Grinold and Kahn, *Active Portfolio Management*;
  - Ang, *Asset Management*.

Chapter numbers for Natenberg, Carver and some others are still to be confirmed before those lessons. The full list of source links goes into `pm-private\library`.

---

## 7. Decisions (2026-10-15, `DECISIONS.md` PM3)

1. The syllabus is approved.
2. The CFA exam window is **August 2027**. Registration opens 3 Nov 2026, and the early price applies until 20 Jan 2027. Registering and paying is the Operator's to do.
3. The read-only account read (the IBKR round) is scheduled in **week 1** (19–25 Oct 2026). It needs its own approved prompt first.
4. This file and `learning-progress.md` were committed once through an `rm/` pull request checked by a fresh agent, in place of the Portfolio Manager Assistant, which is not yet set up.
