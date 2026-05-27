"""Engine realism + reliability verification driver (2026-05-26).

Battery of checks against `origin/main` @ 9f0afaf:
  1. 5-ticker smoke test (CLAUDE.md §4 bring-up)
  2. IV PIT realism: engine's iv column vs Bloomberg IV file directly
  3. EV sign + magnitude consistency check
  4. F4 reproducibility (COST 2022-04-25 prob_profit)
  5. Refusal behaviour during COVID and 2026-03 normal regime

All outputs flat-printed for capture into the verification doc.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKTREE = Path(r"C:\Users\merty\Desktop\swe-terminal-b").resolve()
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

import pandas as pd  # noqa: E402  (sys.path bootstrap above)

from engine.wheel_runner import WheelRunner  # noqa: E402

print("=" * 72)
print("ENGINE REALISM + RELIABILITY VERIFICATION — 2026-05-26")
print("=" * 72)

# --------------------------------------------------------------
# 1. 5-ticker smoke (CLAUDE.md §4)
# --------------------------------------------------------------
print("\n--- (1) 5-ticker smoke ---")
runner = WheelRunner()
print(f"Provider: {type(runner.connector).__name__}")

df = runner.rank_candidates_by_ev(
    tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
    top_n=10,
    min_ev_dollars=-1e9,
    include_diagnostic_fields=True,
)
print(f"Rows returned: {len(df)}")
print(f"Non-null ev_dollars: {df['ev_dollars'].notna().sum()}")
print(f"Non-null iv: {df['iv'].notna().sum()}")
print(f"Non-null premium: {df['premium'].notna().sum()}")
print()
print(df[["ticker", "strike", "premium", "iv", "ev_dollars", "prob_profit"]].to_string(index=False))

# --------------------------------------------------------------
# 2. IV PIT realism — engine iv vs Bloomberg IV file directly
# --------------------------------------------------------------
print("\n\n--- (2) IV PIT realism: engine.iv vs Bloomberg file ---")
iv_file = WORKTREE / "data" / "bloomberg" / "sp500_vol_iv_full.csv"
print(f"IV file: {iv_file} ({iv_file.stat().st_size:,} bytes)")

iv_df = pd.read_csv(iv_file)
iv_df["date"] = pd.to_datetime(iv_df["date"])

as_of_iso = "2026-03-20"
as_of = pd.Timestamp(as_of_iso)


def reference_pit_iv(ticker: str, as_of_ts: pd.Timestamp) -> float | None:
    # File uses Bloomberg ticker format "AAPL UW", "JPM UN" — strip the venue.
    base = iv_df["ticker"].str.split(" ").str[0]
    rows = iv_df[(base == ticker) & (iv_df["date"] <= as_of_ts)]
    if rows.empty:
        return None
    last = rows.iloc[-1]
    put = last.get("hist_put_imp_vol")
    call = last.get("hist_call_imp_vol")
    if pd.isna(put) or pd.isna(call):
        return None
    # The file uses percent (e.g. 28.5). Engine consumes decimal (0.285).
    return float((put + call) / 2.0 / 100.0)


ref_table = []
for t in ["AAPL", "MSFT", "JPM", "XOM", "UNH"]:
    engine_iv = float(df.loc[df["ticker"] == t, "iv"].iloc[0])
    ref_iv = reference_pit_iv(t, as_of)
    if ref_iv is None:
        rel_pct = float("nan")
    else:
        rel_pct = abs(engine_iv - ref_iv) / ref_iv * 100.0
    ref_table.append(
        {
            "ticker": t,
            "engine_iv": round(engine_iv, 6),
            "ref_pit_iv": round(ref_iv, 6) if ref_iv else None,
            "rel_diff_pct": round(rel_pct, 3),
        }
    )
print(f"as_of: {as_of_iso}")
print(pd.DataFrame(ref_table).to_string(index=False))

# --------------------------------------------------------------
# 3. EV sign + magnitude consistency
# --------------------------------------------------------------
print("\n\n--- (3) EV sign + magnitude consistency ---")
print(
    "Expectation: higher IV (rich premium) -> higher ev_dollars at same delta band, modulo regime multipliers."
)
ranked = df.sort_values("iv", ascending=False)
print(ranked[["ticker", "iv", "premium", "ev_dollars", "regime_multiplier"]].to_string(index=False))
print()
print(f"ev_dollars range: {df['ev_dollars'].min():.2f} to {df['ev_dollars'].max():.2f}")
print(
    f"Spearman corr(iv, ev_dollars) = {df[['iv', 'ev_dollars']].corr(method='spearman').iloc[0, 1]:.3f}"
)

# --------------------------------------------------------------
# 4. F4 reproducibility — COST 2022-04-25 prob_profit
# --------------------------------------------------------------
print("\n\n--- (4) F4 reproducibility: COST 2022-04-25 prob_profit ---")
print(
    "Documented in docs/F4_TAIL_RISK_DIAGNOSTIC.md: prob_profit should be 0.8333 at default settings."
)
try:
    df_cost = runner.rank_candidates_by_ev(
        tickers=["COST"],
        as_of="2022-04-25",
        top_n=5,
        min_ev_dollars=-1e9,
        use_event_gate=False,  # F4 reproducibility needs to bypass earnings lockout
        include_diagnostic_fields=True,
    )
    if df_cost.empty:
        drops = df_cost.attrs.get("drops", [])
        print(f"COST 2022-04-25: ranker returned empty; drops={drops}")
    else:
        row = df_cost.iloc[0]
        print(f"COST 2022-04-25 prob_profit = {row['prob_profit']:.6f}")
        print(f"  strike={row['strike']:.2f}, premium={row['premium']:.4f}")
        print(f"  ev_dollars={row['ev_dollars']:.2f}, iv={row['iv']:.6f}")
        print(f"  matches documented 0.8333? {abs(row['prob_profit'] - 0.8333) < 0.001}")
except Exception as exc:
    print(f"F4 reproducibility check FAILED: {exc!r}")

# --------------------------------------------------------------
# 5. Refusal behaviour — COVID + normal regime
# --------------------------------------------------------------
print(
    "\n\n--- (5) Refusal behaviour: COVID 2020-03-23 / mid-COVID 2020-05-11 / normal 2026-03-20 ---"
)
for as_of_check in ["2020-03-23", "2020-05-11", "2026-03-20"]:
    try:
        df_chk = runner.rank_candidates_by_ev(
            tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
            as_of=as_of_check,
            top_n=10,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        drops = df_chk.attrs.get("drops", [])
        n_executed = (df_chk["ev_dollars"] > 0).sum() if not df_chk.empty else 0
        n_total = len(df_chk) + len(drops)
        refusal_pct = 100.0 * (1 - len(df_chk) / max(n_total, 1))
        print(
            f"as_of={as_of_check}: {len(df_chk)} survivors / {n_total} total = {refusal_pct:.1f}% refusal"
        )
        if len(df_chk) > 0:
            print(
                f"  ev_dollars range: {df_chk['ev_dollars'].min():.2f} to {df_chk['ev_dollars'].max():.2f}"
            )
            print(f"  mean regime_multiplier: {df_chk['regime_multiplier'].mean():.3f}")
        else:
            print(f"  drops: {drops[:3]}")
    except Exception as exc:
        print(f"as_of={as_of_check}: FAILED with {exc!r}")

print("\n" + "=" * 72)
print("VERIFICATION COMPLETE")
print("=" * 72)
