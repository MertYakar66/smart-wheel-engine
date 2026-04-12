import math
import re
import sys
import time
import traceback
from datetime import datetime

import requests
from scipy.stats import norm

BASE = "http://localhost:8787"
RESULTS = []
CURRENT_DOMAIN = None
REQUEST_TIMEOUT = 120


def set_domain(name):
    global CURRENT_DOMAIN
    CURRENT_DOMAIN = name
    print(f"\n{name}")


def check(name, condition, expected, actual, severity="MAJOR"):
    status = "PASS" if condition else "FAIL"
    RESULTS.append(
        {
            "domain": CURRENT_DOMAIN,
            "test": name,
            "status": status,
            "expected": expected,
            "actual": actual,
            "severity": severity,
        }
    )
    label = "PASS" if condition else "FAIL"
    print(f"  [{label}] {name}")
    if not condition:
        print(f"       Expected: {expected}")
        print(f"       Actual:   {actual}")
        print(f"       Severity: {severity}")


def fetch(path, **params):
    return requests.get(f"{BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)


def fetch_json(path, **params):
    response = fetch(path, **params)
    try:
        payload = response.json()
    except Exception:
        payload = {"_raw": response.text}
    return response, payload


def as_decimal(value):
    if value is None:
        return None
    return value / 100.0 if value > 1 else value


def as_percent(value):
    if value is None:
        return None
    return value * 100.0 if value <= 1 else value


def parse_sigma(value):
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    cleaned = str(value).replace("σ", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_expected_move_bands(em_payload):
    bands = {}
    for band in em_payload.get("bands", []):
        sigma = parse_sigma(band.get("sigma", band.get("label")))
        if sigma is None:
            continue
        probability = band.get("probability", band.get("probability_within"))
        probability = as_percent(probability) if probability is not None else None
        bands[sigma] = {
            "sigma": sigma,
            "upper": band.get("upper"),
            "lower": band.get("lower"),
            "probability": probability,
            "raw": band,
        }
    return bands


def extract_candidates(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("trades"), list):
            return payload["trades"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
    return []


def ensure_api_up():
    try:
        response = fetch("/api/status")
        check(
            "API status endpoint reachable",
            response.status_code == 200,
            200,
            response.status_code,
            severity="BLOCKER",
        )
        if response.status_code != 200:
            print(response.text[:500])
            return False
        return True
    except Exception as exc:
        check(
            "API status endpoint reachable",
            False,
            "HTTP 200",
            repr(exc),
            severity="BLOCKER",
        )
        return False


def run_domain_1():
    set_domain("D1 OHLCV Integrity")
    ticker_list = ["AAPL", "MSFT", "NVDA", "JPM", "KO"]

    for ticker in ticker_list:
        print(f"\n  Ticker: {ticker}")
        response, chart = fetch_json("/api/chart/ohlcv", ticker=ticker, days=30)
        check(
            f"{ticker}: OHLCV endpoint returns 200",
            response.status_code == 200,
            200,
            response.status_code,
            severity="BLOCKER",
        )

        rows = chart.get("data", []) if isinstance(chart, dict) else []
        check(
            f"{ticker}: OHLCV returns rows",
            len(rows) > 0,
            "> 0 rows",
            len(rows),
            severity="BLOCKER",
        )
        if not rows:
            continue

        row_violations = 0
        parsed_dates = []

        for row in rows:
            date_str = row.get("date")
            fmt_ok = (
                isinstance(date_str, str)
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str) is not None
            )
            check(
                f"OHLCV {ticker} {date_str}: date format YYYY-MM-DD",
                fmt_ok,
                "YYYY-MM-DD",
                date_str,
            )
            if fmt_ok:
                parsed_dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            else:
                row_violations += 1

            low = row.get("low")
            open_ = row.get("open")
            high = row.get("high")
            close = row.get("close")
            volume = row.get("volume")

            checks = [
                ("low <= open <= high", low <= open_ <= high, "ordered OHLC", (low, open_, high)),
                ("low <= close <= high", low <= close <= high, "ordered OHLC", (low, close, high)),
                ("low <= high", low <= high, "low <= high", (low, high)),
                ("volume > 0", volume > 0, "> 0", volume),
            ]
            for label, condition, expected, actual in checks:
                check(f"OHLCV {ticker} {date_str}: {label}", condition, expected, actual)
                if not condition:
                    row_violations += 1

        gap_violations = 0
        for previous, current in zip(parsed_dates, parsed_dates[1:], strict=False):
            gap_days = (current - previous).days
            condition = gap_days <= 5
            check(
                f"OHLCV {ticker}: gap from {previous.date()} to {current.date()} <= 5 days",
                condition,
                "<= 5",
                gap_days,
            )
            if not condition:
                gap_violations += 1

        _, analysis = fetch_json(f"/api/analyze/{ticker}")
        last_close = rows[-1]["close"]
        spot_price = analysis.get("spotPrice") if isinstance(analysis, dict) else None
        check(
            f"{ticker}: spotPrice matches last OHLCV close within $1",
            spot_price is not None and abs(spot_price - last_close) <= 1.0,
            f"{last_close} +/- 1.0",
            spot_price,
        )

        check(f"{ticker}: OHLCV total violations = 0", row_violations == 0, 0, row_violations)
        check(f"{ticker}: OHLCV date gap violations = 0", gap_violations == 0, 0, gap_violations)


def run_domain_2():
    set_domain("D2 Options Math")

    response, r = fetch_json("/api/payoff", ticker="AAPL", strategy="csp", dte=45)
    check("CSP payoff endpoint returns 200", response.status_code == 200, 200, response.status_code)

    premium = r.get("premium")
    strike = r.get("strike")
    spot = r.get("spotPrice", r.get("spot"))
    breakeven = r.get("breakeven")
    max_profit = r.get("maxProfit")

    check("CSP premium > 0", premium > 0, "> 0", premium)
    check("CSP strike < spot", strike < spot, f"< {spot}", strike)

    expected_be = round(strike - premium, 2)
    check(
        "CSP breakeven = strike - premium",
        abs(breakeven - expected_be) < 0.02,
        expected_be,
        breakeven,
    )

    check(
        "CSP maxProfit = premium * 100",
        abs(max_profit - premium * 100) < 0.01,
        premium * 100,
        max_profit,
    )

    data = r.get("data", [])
    prices = [d["price"] for d in data]
    pnls = [d["pnl"] for d in data]

    above_strike = [pnls[i] for i, price in enumerate(prices) if price > strike]
    below_be = [pnls[i] for i, price in enumerate(prices) if price < breakeven - 1]
    at_strike = [pnls[i] for i, price in enumerate(prices) if abs(price - strike) < 0.5]
    at_be = [pnls[i] for i, price in enumerate(prices) if abs(price - breakeven) < 1.0]

    check(
        "CSP: pnl flat at maxProfit above strike",
        bool(above_strike) and all(abs(value - max_profit) < 1.0 for value in above_strike),
        f"all ~= {max_profit}",
        above_strike[:3],
    )

    check(
        "CSP: pnl negative below breakeven",
        bool(below_be) and all(value < 0 for value in below_be),
        "all < 0",
        below_be[:3],
    )

    if at_strike:
        check(
            "CSP: pnl ~= maxProfit at strike",
            abs(at_strike[0] - max_profit) < 5,
            max_profit,
            at_strike[0],
        )

    if at_be:
        check(
            "CSP: pnl ~= $0 at breakeven",
            abs(at_be[0]) < 10,
            "~= 0",
            at_be[0],
        )

    cc_response, cc = fetch_json("/api/payoff", ticker="AAPL", strategy="cc", dte=45)
    check(
        "CC payoff endpoint returns 200",
        cc_response.status_code == 200,
        200,
        cc_response.status_code,
    )

    cc_data = cc.get("data", [])
    cc_prices = [d["price"] for d in cc_data]
    cc_pnls = [d["pnl"] for d in cc_data]

    above_cc_strike = [
        cc_pnls[i] for i, price in enumerate(cc_prices) if price > cc["strike"] * 1.02
    ]
    if len(above_cc_strike) >= 3:
        variance = max(above_cc_strike) - min(above_cc_strike)
        check(
            "CC: profit capped above strike (flat pnl)",
            variance < 5.0,
            "variance < $5",
            variance,
        )

    check(
        "CC: breakeven < spot",
        cc["breakeven"] < cc.get("spotPrice", cc.get("spot")),
        f"< {cc.get('spotPrice', cc.get('spot'))}",
        cc["breakeven"],
    )

    em_response, em = fetch_json("/api/expected_move", ticker="AAPL", dte=45)
    check(
        "expected_move endpoint returns 200",
        em_response.status_code == 200,
        200,
        em_response.status_code,
    )

    spot = em.get("spot")
    iv = as_decimal(em.get("iv"))
    dte = 45
    period_vol = as_decimal(em.get("period_vol"))

    expected_pv = iv * math.sqrt(dte / 365)
    check(
        "period_vol = IV * sqrt(DTE/365)",
        abs(period_vol - expected_pv) < 0.001,
        round(expected_pv, 6),
        period_vol,
    )

    bands = normalize_expected_move_bands(em)
    check("Expected move includes 1sigma band", 1.0 in bands, "band exists", sorted(bands))
    check("Expected move includes 1.5sigma band", 1.5 in bands, "band exists", sorted(bands))
    check("Expected move includes 2sigma band", 2.0 in bands, "band exists", sorted(bands))

    if 1.0 in bands:
        check(
            "1sigma probability ~= 68.3%",
            abs(bands[1.0]["probability"] - 68.27) < 1.0,
            "68.27 +/- 1.0",
            bands[1.0]["probability"],
        )

    if 1.5 in bands:
        expected_prob = (norm.cdf(1.5) - norm.cdf(-1.5)) * 100
        check(
            "1.5sigma probability ~= 86.6%",
            abs(bands[1.5]["probability"] - expected_prob) < 1.0,
            f"{expected_prob:.2f} +/- 1.0",
            bands[1.5]["probability"],
        )

    if 2.0 in bands:
        check(
            "2sigma probability ~= 95.4%",
            abs(bands[2.0]["probability"] - 95.45) < 1.0,
            "95.45 +/- 1.0",
            bands[2.0]["probability"],
        )

    for sigma, band in bands.items():
        expected_width = 2 * spot * period_vol * sigma
        actual_width = band["upper"] - band["lower"]
        check(
            f"{sigma}sigma band width = 2*spot*pv*sigma",
            expected_width > 0 and abs(actual_width - expected_width) / expected_width < 0.01,
            round(expected_width, 2),
            round(actual_width, 2),
        )

    _, csp_strikes = fetch_json("/api/strikes", ticker="AAPL", strategy="csp", dte=45)
    recs = csp_strikes.get("recommendations", [])
    _, analysis = fetch_json("/api/analyze/AAPL")
    spot_s = analysis.get("spotPrice")

    for rec in recs:
        check(
            f"CSP strike {rec['strike']} < spot",
            rec["strike"] < spot_s,
            f"< {spot_s}",
            rec["strike"],
        )
        check(
            f"CSP delta {rec['delta']} is negative",
            rec["delta"] < 0,
            "< 0",
            rec["delta"],
        )
        check(
            "CSP probOTM in [50,95]",
            50 <= rec["probabilityOtm"] <= 95,
            "[50, 95]",
            rec["probabilityOtm"],
        )
        check(
            "CSP breakeven = strike - premium",
            abs(rec["breakeven"] - (rec["strike"] - rec["premium"])) < 0.02,
            round(rec["strike"] - rec["premium"], 2),
            rec["breakeven"],
        )

    scores = [item["score"] for item in recs]
    check(
        "CSP recs sorted by score DESC",
        scores == sorted(scores, reverse=True),
        "descending",
        scores,
    )

    _, cc_strikes = fetch_json("/api/strikes", ticker="AAPL", strategy="cc", dte=45)
    cc_recs = cc_strikes.get("recommendations", [])

    above_spot = sum(1 for rec in cc_recs if rec["strike"] > spot_s)
    check(
        "CC: most strikes above spot (OTM calls)",
        above_spot >= len(cc_recs) * 0.6,
        f">= {int(len(cc_recs) * 0.6)} above spot",
        above_spot,
    )

    for rec in cc_recs:
        check(
            f"CC delta {rec['delta']} is positive",
            rec["delta"] > 0,
            "> 0",
            rec["delta"],
        )

    if len(cc_recs) >= 2:
        delta_premium = [(rec["delta"], rec["premium"]) for rec in cc_recs]
        delta_premium.sort(key=lambda item: item[0])
        premiums_sorted = [premium for _, premium in delta_premium]
        is_monotonic = all(
            premiums_sorted[i] <= premiums_sorted[i + 1] for i in range(len(premiums_sorted) - 1)
        )
        check(
            "CC: higher delta -> higher premium",
            is_monotonic,
            "monotonic",
            premiums_sorted,
        )

    _, raw_candidates = fetch_json("/api/candidates", limit=5, min_score=50)
    cands = extract_candidates(raw_candidates)
    check("Candidates endpoint returned 5 or fewer trades", len(cands) <= 5, "<= 5", len(cands))

    for candidate in cands:
        ticker = candidate["ticker"]
        reported_spot = candidate.get("spot", candidate.get("spotPrice"))
        check(
            f"{ticker}: candidate includes spot field",
            reported_spot is not None,
            "spot present",
            reported_spot,
        )

        if reported_spot is None:
            _, ticker_analysis = fetch_json(f"/api/analyze/{ticker}")
            effective_spot = ticker_analysis.get("spotPrice")
        else:
            effective_spot = reported_spot

        check(
            f"{ticker}: delta in [-0.50, -0.10]",
            -0.50 <= candidate["delta"] <= -0.10,
            "[-0.50, -0.10]",
            candidate["delta"],
        )
        check(f"{ticker}: premium > 0", candidate["premium"] > 0, "> 0", candidate["premium"])
        check(
            f"{ticker}: strike > 0 and < spot",
            effective_spot is not None and 0 < candidate["strike"] < effective_spot,
            f"(0, {effective_spot})",
            candidate["strike"],
        )
        check(
            f"{ticker}: probability > 50",
            candidate["probability"] > 50,
            "> 50",
            candidate["probability"],
        )
        check(
            f"{ticker}: expectedPnL > 0",
            candidate["expectedPnL"] > 0,
            "> 0",
            candidate["expectedPnL"],
        )


def run_domain_3():
    set_domain("D3 Committee Quality")
    committee_tickers = ["NVDA", "KO", "AAPL"]

    for ticker in committee_tickers:
        print(f"\n  Committee: {ticker}")
        response, committee = fetch_json("/api/committee", ticker=ticker)
        check(
            f"{ticker}: committee endpoint returns 200",
            response.status_code == 200,
            200,
            response.status_code,
            severity="BLOCKER",
        )
        advisors = committee.get("advisors", [])
        names = [advisor.get("name") for advisor in advisors]

        check(f"{ticker}: exactly 4 advisors", len(advisors) == 4, 4, len(advisors))
        check(
            f"{ticker}: all 4 present (Buffett/Munger/Simons/Taleb)",
            set(names) == {"Warren Buffett", "Charlie Munger", "Jim Simons", "Nassim Taleb"},
            "all 4",
            names,
        )

        for advisor in advisors:
            name = advisor.get("name")
            check(
                f"{ticker}/{name}: >=2 keyReasons",
                len(advisor.get("keyReasons", [])) >= 2,
                ">=2",
                len(advisor.get("keyReasons", [])),
            )
            check(
                f"{ticker}/{name}: >=1 criticalQuestions",
                len(advisor.get("criticalQuestions", [])) >= 1,
                ">=1",
                len(advisor.get("criticalQuestions", [])),
            )
            check(
                f"{ticker}/{name}: >=1 hiddenRisks",
                len(advisor.get("hiddenRisks", [])) >= 1,
                ">=1",
                len(advisor.get("hiddenRisks", [])),
            )

            if "Taleb" in str(name):
                check(
                    f"{ticker}/Taleb: NOT strong_approve",
                    advisor.get("judgment") != "strong_approve",
                    "not strong_approve",
                    advisor.get("judgment"),
                )

        _, analysis = fetch_json(f"/api/analyze/{ticker}")
        spot_c = analysis.get("spotPrice")
        trade = committee.get("trade", {})
        check(
            f"{ticker}: committee response includes trade block",
            bool(trade),
            "trade present",
            trade,
        )

        trade_strike = trade.get("strike", 0) if isinstance(trade, dict) else 0
        pct_below = (spot_c - trade_strike) / spot_c if spot_c else 0
        check(
            f"{ticker}: strike ~8% below spot (4-15% range)",
            0.04 <= pct_below <= 0.15,
            "4-15% below spot",
            f"{pct_below:.1%}",
        )

        report_len = len(committee.get("report", ""))
        check(f"{ticker}: report > 500 chars", report_len > 500, "> 500", report_len)

    for ticker in committee_tickers:
        _, committee = fetch_json("/api/committee", ticker=ticker)
        all_reasons = []
        for advisor in committee.get("advisors", []):
            all_reasons.extend(advisor.get("keyReasons", []))
        unique = len(set(all_reasons))
        total = len(all_reasons)
        check(
            f"{ticker}: all keyReasons unique across advisors",
            unique == total,
            f"all {total} unique",
            f"{unique}/{total} unique",
        )

    r400 = fetch("/api/committee", ticker="")
    check(
        "Empty ticker -> 400",
        r400.status_code == 400,
        "HTTP 400",
        r400.status_code,
        severity="BLOCKER",
    )


def run_domain_4():
    set_domain("D4 Strangle Timing")
    _, r = fetch_json("/api/strangle", ticker="AAPL")

    check("strangle score in [0,100]", 0 <= r["score"] <= 100, "[0,100]", r["score"])
    check(
        "strangle phase is valid",
        r["phase"] in ["compression", "expansion", "post_expansion", "trend", "unknown"],
        "valid phase",
        r["phase"],
    )

    required_components = ["bollinger", "atr", "rsi", "trend", "range"]
    component_keys = list(r.get("components", {}).keys())
    check(
        "exactly 5 components",
        set(component_keys) == set(required_components),
        required_components,
        component_keys,
    )

    for component_name in required_components:
        component = r["components"][component_name]
        check(
            f"{component_name}: score in [0,100]",
            0 <= component["score"] <= 100,
            "[0,100]",
            component["score"],
        )
        check(
            f"{component_name}: state is string",
            isinstance(component["state"], str),
            "str",
            type(component["state"]).__name__,
        )

    metrics = r.get("metrics", {})
    check(
        "rsi_14 in [0,100]",
        0 <= metrics.get("rsi_14", -1) <= 100,
        "[0,100]",
        metrics.get("rsi_14"),
    )
    check(
        "bb_width_pctl in [0,100]",
        0 <= metrics.get("bb_width_pctl", -1) <= 100,
        "[0,100]",
        metrics.get("bb_width_pctl"),
    )

    warnings = r.get("warnings", {})
    check(
        "warnings.compression is bool",
        isinstance(warnings.get("compression"), bool),
        "bool",
        type(warnings.get("compression")).__name__,
    )

    scores_3 = {}
    for ticker in ["AAPL", "MSFT", "JPM"]:
        _, payload = fetch_json("/api/strangle", ticker=ticker)
        scores_3[ticker] = payload["score"]
    check(
        "3 tickers have different strangle scores (not hardcoded)",
        len(set(scores_3.values())) > 1,
        "different values",
        scores_3,
    )


def run_domain_5():
    set_domain("D5 Robustness")

    r404 = fetch("/api/analyze/ZZZZ")
    check("ZZZZ -> HTTP 404", r404.status_code == 404, 404, r404.status_code, "BLOCKER")
    check(
        "ZZZZ -> has error field",
        "error" in r404.json(),
        "error field present",
        list(r404.json().keys()),
    )

    r_default = fetch("/api/analyze/")
    check(
        "/api/analyze/ -> 200 (AAPL default)",
        r_default.status_code == 200,
        200,
        r_default.status_code,
    )

    r_bad_chart = fetch("/api/chart/invalid_type", ticker="AAPL")
    check("invalid chart type -> 400", r_bad_chart.status_code == 400, 400, r_bad_chart.status_code)

    r_butterfly = fetch("/api/strikes", ticker="AAPL", strategy="butterfly")
    check(
        "butterfly strategy -> empty recs (not crash)",
        r_butterfly.status_code == 200,
        200,
        r_butterfly.status_code,
    )

    r5 = fetch("/api/chart/bollinger", ticker="AAPL", days=5)
    data5 = r5.json().get("data", [])
    check("days=5 returns exactly 5 rows", len(data5) == 5, 5, len(data5))

    r2000 = fetch("/api/chart/bollinger", ticker="AAPL", days=2000)
    check("days=2000 returns data (no crash)", r2000.status_code == 200, 200, r2000.status_code)

    r_dte1 = fetch("/api/expected_move", ticker="AAPL", dte=1)
    check("dte=1 expected move works", r_dte1.status_code == 200, 200, r_dte1.status_code)

    r_dte365 = fetch("/api/expected_move", ticker="AAPL", dte=365)
    check("dte=365 expected move works", r_dte365.status_code == 200, 200, r_dte365.status_code)

    times = []
    for _ in range(3):
        start = time.time()
        fetch("/api/analyze/AAPL")
        times.append(time.time() - start)
    avg_ms = sum(times) / len(times) * 1000
    check("analyze/AAPL avg response < 3s", avg_ms < 3000, "< 3000ms", f"{avg_ms:.0f}ms")


def print_report():
    print("\n" + "=" * 60)
    print("SMART WHEEL ENGINE - BACKEND AUDIT RESULTS")
    print("=" * 60)

    ordered_domains = [
        "D1 OHLCV Integrity",
        "D2 Options Math",
        "D3 Committee Quality",
        "D4 Strangle Timing",
        "D5 Robustness",
    ]

    total_pass = 0
    total_fail = 0

    for domain in ordered_domains:
        tests = [result for result in RESULTS if result["domain"] == domain]
        passed = sum(1 for test in tests if test["status"] == "PASS")
        failed = sum(1 for test in tests if test["status"] == "FAIL")
        total_pass += passed
        total_fail += failed
        rating = (
            "PRODUCTION READY" if failed == 0 else ("NEAR READY" if failed <= 2 else "NOT READY")
        )
        print(f"\n{domain}: {passed}/{passed + failed} [{rating}]")
        for test in tests:
            if test["status"] == "FAIL":
                print(f"  FAIL [{test['severity']}] {test['test']}")
                print(f"       Expected: {test['expected']}")
                print(f"       Actual:   {test['actual']}")

    total = total_pass + total_fail
    pct = total_pass / total * 100 if total else 0
    verdict = "READY" if pct >= 90 else ("CONDITIONAL" if pct >= 75 else "NOT READY")

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_pass}/{total} ({pct:.1f}%)")
    print(f"LAUNCH RECOMMENDATION: {verdict}")
    print("=" * 60)


def main():
    if not ensure_api_up():
        print_report()
        sys.exit(1)

    # Warm up compute-heavy endpoints so the first-call cold start doesn't
    # blow through the per-request timeout inside the domain runners.
    print("\n  Warming up /api/candidates and /api/committee ...")
    try:
        fetch("/api/candidates", limit=5, min_score=50)
        fetch("/api/committee", ticker="AAPL")
    except Exception as exc:
        print(f"  warmup warning: {exc}")

    run_domain_1()
    run_domain_2()
    run_domain_3()
    run_domain_4()
    run_domain_5()
    print_report()

    blocker_failures = [r for r in RESULTS if r["status"] == "FAIL" and r["severity"] == "BLOCKER"]
    sys.exit(1 if blocker_failures else 0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nUNEXPECTED AUDIT ERROR")
        traceback.print_exc()
        print_report()
        sys.exit(1)
