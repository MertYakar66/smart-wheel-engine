# Premium-correction pilot — summary

Total rows: 200  |  clean joins: 200  |  resolved (terminal price known): 200

HEADLINE (Refinement 1) = premium under-pricing (real_mid − BSM(iv));
skew-driven, NOT VRP.

## AAPL  (n=120)
  correction $/contract : median +56.2  IQR [+7.6, +77.2]
  correction % of BSM   : median +19.4%  IQR [+4.9%, +36.5%]

## NVDA  (n=25)
  correction $/contract : median +10.0  IQR [-4.3, +25.3]
  correction % of BSM   : median +3.4%  IQR [-1.8%, +7.9%]

## TSLA  (n=55)
  correction $/contract : median +1.6  IQR [-24.3, +37.6]
  correction % of BSM   : median +0.2%  IQR [-2.2%, +6.9%]

## Refinement 2 — engine predicted vs REALIZED assignment (physical-vs-physical)
  overall: predicted 0.253 vs realized 0.165 (gap -0.088) — but n=200 contracts come from only n_clusters=14 independent (ticker, as_of) events: NOT 155 trials.
  per-name gap (realized − predicted):
    AAPL: pred 0.253  real 0.275  gap +0.022  (n=120, clusters=9)
    NVDA: pred 0.159  real 0.000  gap -0.159  (n=25, clusters=2)
    TSLA: pred 0.295  real 0.000  gap -0.295  (n=55, clusters=3)

  miscalibration vs premium correction (cluster-robust CI):
    correction%    n  clus  predicted  realized [cluster95]    gap[cluster95]      flag
       -9.2%    34   11   0.286     0.147 [0.000,0.423]  -0.139[-0.285,+0.129]
       -1.1%    33    7   0.284     0.030 [0.000,0.107]  -0.254[-0.307,-0.152]  <not signal: <8 clusters or <30 n>
       +4.4%    33    8   0.242     0.182 [0.038,0.444]  -0.060[-0.223,+0.189]
      +12.8%    33    7   0.248     0.091 [0.000,0.381]  -0.157[-0.267,+0.172]  <not signal: <8 clusters or <30 n>
      +25.0%    33    6   0.247     0.273 [0.000,0.711]  +0.026[-0.284,+0.477]  <not signal: <8 clusters or <30 n>
      +57.7%    34    5   0.212     0.265 [0.000,0.818]  +0.053[-0.249,+0.609]  <not signal: <8 clusters or <30 n>

    READ (trustworthy bins, n>=30 AND clusters>=8): high-corr gap cluster-robust CI does NOT clear 0 ⇒ NO calibration failure established; the apparent signal is within cluster-robust noise.

  NOTE: `risk_premium_wedge` (Q − P) is in the records for context only — it is the risk premium, NOT a calibration gap, and is not used as a deliverable axis.
  CAVEAT: calm 2024–25 band; the crisis-onset under-seeing case needs a 2020/2022 stress window (deferred to the full study).