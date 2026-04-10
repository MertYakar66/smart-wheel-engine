"""
Portfolio Intelligence — Congress & Institutional Tracking with Overlap Radar

Tracks and analyzes trading activity from:
- US Congress members (SEC Periodic Transaction Reports)
- Major institutional investors (13F quarterly filings)
- Cross-references with user's portfolio/watchlist for overlap detection

Data sources:
- Congress: SEC PTR filings via CSV/JSON (CapitolTrades, Quiver Quantitative)
- Institutional: SEC EDGAR 13F filings via CSV/JSON
- User portfolio: from engine's PortfolioTracker or manual input

Usage:
    from engine.portfolio_intelligence import CongressTracker, InstitutionalTracker, OverlapRadar

    congress = CongressTracker()
    congress.load_filings("data/congress/filings.csv")
    recent = congress.get_recent_trades(days=30)
    clusters = congress.get_clustering(days=60, min_members=3)

    institutional = InstitutionalTracker()
    institutional.load_holdings("data/institutional/13f_latest.csv")
    conviction = institutional.get_conviction_holdings(min_portfolio_pct=5.0)

    radar = OverlapRadar(congress, institutional)
    overlaps = radar.analyze_overlaps(["AAPL", "MSFT", "JPM"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Amount range midpoints for Congress filings
AMOUNT_MIDPOINTS = {
    "$1,001 - $15,000": 8_000,
    "$15,001 - $50,000": 32_500,
    "$50,001 - $100,000": 75_000,
    "$100,001 - $250,000": 175_000,
    "$250,001 - $500,000": 375_000,
    "$500,001 - $1,000,000": 750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
    "$5,000,001 - $25,000,000": 15_000_000,
    "$25,000,001 - $50,000,000": 37_500_000,
    "Over $50,000,000": 75_000_000,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CongressFiling:
    """A single Congressional trading disclosure."""

    member_name: str
    member_party: str  # "D", "R", "I"
    member_chamber: str  # "Senate", "House"
    ticker: str
    transaction_type: str  # "Purchase", "Sale", "Exchange"
    amount_range: str
    amount_estimate: float  # Midpoint of range
    transaction_date: date
    disclosure_date: date
    disclosure_delay_days: int
    sector: str = ""
    asset_type: str = "Stock"

    def to_dict(self) -> dict:
        return {
            "member_name": self.member_name,
            "member_party": self.member_party,
            "member_chamber": self.member_chamber,
            "ticker": self.ticker,
            "transaction_type": self.transaction_type,
            "amount_range": self.amount_range,
            "amount_estimate": self.amount_estimate,
            "transaction_date": self.transaction_date.isoformat(),
            "disclosure_date": self.disclosure_date.isoformat(),
            "disclosure_delay_days": self.disclosure_delay_days,
            "sector": self.sector,
            "asset_type": self.asset_type,
        }


@dataclass
class InstitutionalHolding:
    """A single institutional fund holding from 13F filing."""

    fund_name: str
    ticker: str
    shares: int
    value: float
    portfolio_pct: float
    change_shares: int
    change_pct: float
    action: str  # "New", "Increased", "Decreased", "Sold", "Unchanged"
    filing_date: date
    quarter: str

    def to_dict(self) -> dict:
        return {
            "fund_name": self.fund_name,
            "ticker": self.ticker,
            "shares": self.shares,
            "value": self.value,
            "portfolio_pct": self.portfolio_pct,
            "change_shares": self.change_shares,
            "change_pct": self.change_pct,
            "action": self.action,
            "filing_date": self.filing_date.isoformat(),
            "quarter": self.quarter,
        }


# =============================================================================
# Congress Tracker
# =============================================================================


class CongressTracker:
    """Track and analyze Congressional trading activity."""

    def __init__(self, data_dir: str | Path = "data/congress"):
        self.data_dir = Path(data_dir)
        self.filings: list[CongressFiling] = []

    def load_filings(self, filepath: str | Path) -> int:
        """
        Load filings from CSV.

        Expected columns: member_name, member_party, member_chamber, ticker,
        transaction_type, amount_range, transaction_date, disclosure_date, sector

        Returns number of filings loaded.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Congress filings not found: {filepath}")
            return 0

        df = pd.read_csv(filepath)
        count = 0
        for _, row in df.iterrows():
            try:
                txn_date = pd.Timestamp(row["transaction_date"]).date()
                disc_date = pd.Timestamp(row.get("disclosure_date", row["transaction_date"])).date()
                amount_range = row.get("amount_range", "$1,001 - $15,000")
                filing = CongressFiling(
                    member_name=str(row["member_name"]),
                    member_party=str(row.get("member_party", "?")),
                    member_chamber=str(row.get("member_chamber", "?")),
                    ticker=str(row["ticker"]).upper().strip(),
                    transaction_type=str(row.get("transaction_type", "Purchase")),
                    amount_range=amount_range,
                    amount_estimate=AMOUNT_MIDPOINTS.get(amount_range, 8_000),
                    transaction_date=txn_date,
                    disclosure_date=disc_date,
                    disclosure_delay_days=(disc_date - txn_date).days,
                    sector=str(row.get("sector", "")),
                    asset_type=str(row.get("asset_type", "Stock")),
                )
                self.filings.append(filing)
                count += 1
            except Exception:
                continue
        return count

    def add_filing(self, filing: CongressFiling) -> None:
        self.filings.append(filing)

    def get_recent_trades(self, days: int = 30) -> list[CongressFiling]:
        cutoff = date.today() - timedelta(days=days)
        return sorted(
            [f for f in self.filings if f.transaction_date >= cutoff],
            key=lambda f: f.transaction_date,
            reverse=True,
        )

    def get_trades_by_ticker(self, ticker: str) -> list[CongressFiling]:
        return sorted(
            [f for f in self.filings if f.ticker == ticker.upper()],
            key=lambda f: f.transaction_date,
            reverse=True,
        )

    def get_trades_by_member(self, member: str) -> list[CongressFiling]:
        member_lower = member.lower()
        return sorted(
            [f for f in self.filings if member_lower in f.member_name.lower()],
            key=lambda f: f.transaction_date,
            reverse=True,
        )

    def get_clustering(self, days: int = 60, min_members: int = 3) -> pd.DataFrame:
        """Find tickers traded by multiple members in a window."""
        cutoff = date.today() - timedelta(days=days)
        recent = [f for f in self.filings if f.transaction_date >= cutoff]

        ticker_data: dict[str, dict] = {}
        for f in recent:
            if f.ticker not in ticker_data:
                ticker_data[f.ticker] = {
                    "ticker": f.ticker,
                    "buy_count": 0,
                    "sell_count": 0,
                    "members": set(),
                    "total_est_value": 0,
                    "sector": f.sector,
                }
            td = ticker_data[f.ticker]
            if f.transaction_type.lower() in ("purchase", "buy"):
                td["buy_count"] += 1
            else:
                td["sell_count"] += 1
            td["members"].add(f.member_name)
            td["total_est_value"] += f.amount_estimate

        rows = []
        for td in ticker_data.values():
            if len(td["members"]) >= min_members:
                rows.append(
                    {
                        "ticker": td["ticker"],
                        "unique_members": len(td["members"]),
                        "buy_count": td["buy_count"],
                        "sell_count": td["sell_count"],
                        "net_direction": "Buy" if td["buy_count"] > td["sell_count"] else "Sell",
                        "total_est_value": td["total_est_value"],
                        "sector": td["sector"],
                        "members": ", ".join(sorted(td["members"])),
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("unique_members", ascending=False)
        return df

    def get_sector_activity(self, days: int = 90) -> pd.DataFrame:
        cutoff = date.today() - timedelta(days=days)
        recent = [f for f in self.filings if f.transaction_date >= cutoff and f.sector]

        sector_data: dict[str, dict] = {}
        for f in recent:
            s = f.sector
            if s not in sector_data:
                sector_data[s] = {"sector": s, "buys": 0, "sells": 0, "total_value": 0}
            if f.transaction_type.lower() in ("purchase", "buy"):
                sector_data[s]["buys"] += 1
            else:
                sector_data[s]["sells"] += 1
            sector_data[s]["total_value"] += f.amount_estimate

        return pd.DataFrame(sector_data.values()).sort_values("total_value", ascending=False)

    def get_disclosure_delay_stats(self) -> dict:
        if not self.filings:
            return {"mean": 0, "median": 0, "max": 0, "p90": 0}
        delays = [f.disclosure_delay_days for f in self.filings]
        import numpy as np

        return {
            "mean": float(np.mean(delays)),
            "median": float(np.median(delays)),
            "max": int(np.max(delays)),
            "p90": float(np.percentile(delays, 90)),
        }


# =============================================================================
# Institutional (13F) Tracker
# =============================================================================


TRACKED_FUNDS = {
    "Berkshire Hathaway": {"cik": "0001067983", "manager": "Warren Buffett"},
    "Pershing Square": {"cik": "0001336528", "manager": "Bill Ackman"},
    "Greenlight Capital": {"cik": "0001079114", "manager": "David Einhorn"},
    "Scion Asset Management": {"cik": "0001649339", "manager": "Michael Burry"},
    "Third Point": {"cik": "0001040273", "manager": "Dan Loeb"},
    "Appaloosa Management": {"cik": "0001656456", "manager": "David Tepper"},
}


class InstitutionalTracker:
    """Track major institutional investor portfolios from 13F filings."""

    def __init__(self, data_dir: str | Path = "data/institutional"):
        self.data_dir = Path(data_dir)
        self.holdings: list[InstitutionalHolding] = []

    def load_holdings(self, filepath: str | Path) -> int:
        """
        Load holdings from CSV.

        Expected columns: fund_name, ticker, shares, value, portfolio_pct,
        change_shares, change_pct, action, filing_date, quarter
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Institutional holdings not found: {filepath}")
            return 0

        df = pd.read_csv(filepath)
        count = 0
        for _, row in df.iterrows():
            try:
                holding = InstitutionalHolding(
                    fund_name=str(row["fund_name"]),
                    ticker=str(row["ticker"]).upper().strip(),
                    shares=int(row.get("shares", 0)),
                    value=float(row.get("value", 0)),
                    portfolio_pct=float(row.get("portfolio_pct", 0)),
                    change_shares=int(row.get("change_shares", 0)),
                    change_pct=float(row.get("change_pct", 0)),
                    action=str(row.get("action", "Unchanged")),
                    filing_date=pd.Timestamp(row.get("filing_date", "2025-01-01")).date(),
                    quarter=str(row.get("quarter", "")),
                )
                self.holdings.append(holding)
                count += 1
            except Exception:
                continue
        return count

    def add_holding(self, holding: InstitutionalHolding) -> None:
        self.holdings.append(holding)

    def get_fund_holdings(
        self, fund_name: str, quarter: str | None = None
    ) -> list[InstitutionalHolding]:
        result = [h for h in self.holdings if h.fund_name == fund_name]
        if quarter:
            result = [h for h in result if h.quarter == quarter]
        return sorted(result, key=lambda h: h.portfolio_pct, reverse=True)

    def get_new_positions(self, quarter: str | None = None) -> pd.DataFrame:
        filtered = self.holdings
        if quarter:
            filtered = [h for h in filtered if h.quarter == quarter]
        new = [h for h in filtered if h.action == "New"]
        return pd.DataFrame([h.to_dict() for h in new])

    def get_increased_positions(self, quarter: str | None = None) -> pd.DataFrame:
        filtered = self.holdings
        if quarter:
            filtered = [h for h in filtered if h.quarter == quarter]
        increased = [h for h in filtered if h.action in ("New", "Increased")]
        return pd.DataFrame([h.to_dict() for h in increased])

    def get_conviction_holdings(self, min_portfolio_pct: float = 5.0) -> pd.DataFrame:
        conviction = [h for h in self.holdings if h.portfolio_pct >= min_portfolio_pct]
        return (
            pd.DataFrame([h.to_dict() for h in conviction]).sort_values(
                "portfolio_pct", ascending=False
            )
            if conviction
            else pd.DataFrame()
        )

    def get_multi_fund_overlap(self, min_funds: int = 2) -> pd.DataFrame:
        ticker_funds: dict[str, set[str]] = {}
        ticker_value: dict[str, float] = {}
        for h in self.holdings:
            ticker_funds.setdefault(h.ticker, set()).add(h.fund_name)
            ticker_value[h.ticker] = ticker_value.get(h.ticker, 0) + h.value

        rows = []
        for ticker, funds in ticker_funds.items():
            if len(funds) >= min_funds:
                rows.append(
                    {
                        "ticker": ticker,
                        "fund_count": len(funds),
                        "funds": ", ".join(sorted(funds)),
                        "total_value": ticker_value.get(ticker, 0),
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("fund_count", ascending=False)
        return df

    def get_available_funds(self) -> list[str]:
        return sorted({h.fund_name for h in self.holdings})


# =============================================================================
# Overlap Radar
# =============================================================================


class OverlapRadar:
    """Cross-reference tracked portfolios with user's watchlist/portfolio."""

    def __init__(
        self,
        congress: CongressTracker | None = None,
        institutional: InstitutionalTracker | None = None,
    ):
        self.congress = congress or CongressTracker()
        self.institutional = institutional or InstitutionalTracker()

    def analyze_overlaps(self, user_tickers: list[str], days: int = 90) -> dict:
        """
        Full overlap analysis between user portfolio and tracked sources.

        Returns:
            Dict with congress_overlap, institutional_overlap,
            smart_money_consensus, new_ideas
        """
        user_set = {t.upper() for t in user_tickers}

        # Congress overlap
        congress_tickers: dict[str, list[str]] = {}
        recent_congress = self.congress.get_recent_trades(days=days)
        for f in recent_congress:
            if f.ticker in user_set:
                congress_tickers.setdefault(f.ticker, []).append(
                    f"{f.member_name} ({f.transaction_type})"
                )

        # Institutional overlap
        inst_tickers: dict[str, list[str]] = {}
        for h in self.institutional.holdings:
            if h.ticker in user_set:
                inst_tickers.setdefault(h.ticker, []).append(
                    f"{h.fund_name} ({h.action}, {h.portfolio_pct:.1f}%)"
                )

        # Smart money consensus (in both Congress AND institutional)
        congress_all = {f.ticker for f in recent_congress}
        inst_all = {h.ticker for h in self.institutional.holdings}
        consensus = congress_all & inst_all

        # New ideas (in tracked sources but NOT in user portfolio)
        new_ideas_congress = congress_all - user_set
        new_ideas_inst = inst_all - user_set
        new_ideas_both = new_ideas_congress & new_ideas_inst

        return {
            "congress_overlap": dict(sorted(congress_tickers.items())),
            "institutional_overlap": dict(sorted(inst_tickers.items())),
            "smart_money_consensus": sorted(consensus),
            "new_ideas": {
                "in_both_sources": sorted(new_ideas_both)[:20],
                "congress_only": sorted(new_ideas_congress - new_ideas_inst)[:20],
                "institutional_only": sorted(new_ideas_inst - new_ideas_congress)[:20],
            },
            "user_tickers_analyzed": len(user_set),
            "congress_filings_scanned": len(recent_congress),
            "institutional_holdings_scanned": len(self.institutional.holdings),
        }

    def get_ticker_intelligence(self, ticker: str) -> dict:
        """Everything we know about a ticker from tracked sources."""
        ticker = ticker.upper()

        congress_trades = self.congress.get_trades_by_ticker(ticker)
        inst_holders = [h for h in self.institutional.holdings if h.ticker == ticker]

        congress_buys = sum(
            1 for f in congress_trades if f.transaction_type.lower() in ("purchase", "buy")
        )
        congress_sells = sum(
            1 for f in congress_trades if f.transaction_type.lower() in ("sale", "sell")
        )

        # Consensus score: +1 for each buy signal, -1 for each sell
        score = congress_buys - congress_sells + len(inst_holders) * 2

        return {
            "ticker": ticker,
            "consensus_score": score,
            "congress": {
                "total_trades": len(congress_trades),
                "buys": congress_buys,
                "sells": congress_sells,
                "members": list({f.member_name for f in congress_trades}),
                "recent": [f.to_dict() for f in congress_trades[:5]],
            },
            "institutional": {
                "fund_count": len(inst_holders),
                "funds": [h.fund_name for h in inst_holders],
                "total_value": sum(h.value for h in inst_holders),
                "holdings": [h.to_dict() for h in inst_holders],
            },
        }

    def generate_report(self, user_tickers: list[str], days: int = 90) -> str:
        """Formatted intelligence report."""
        overlaps = self.analyze_overlaps(user_tickers, days)
        lines = [
            "=" * 60,
            "PORTFOLIO INTELLIGENCE REPORT",
            "=" * 60,
            "",
            f"User tickers analyzed: {overlaps['user_tickers_analyzed']}",
            f"Congress filings scanned: {overlaps['congress_filings_scanned']}",
            f"Institutional holdings scanned: {overlaps['institutional_holdings_scanned']}",
            "",
        ]

        # Congress overlap
        if overlaps["congress_overlap"]:
            lines.append("-" * 40)
            lines.append("CONGRESS OVERLAP (your holdings traded by Congress)")
            lines.append("-" * 40)
            for ticker, members in overlaps["congress_overlap"].items():
                lines.append(f"  {ticker}: {', '.join(members[:3])}")
            lines.append("")

        # Institutional overlap
        if overlaps["institutional_overlap"]:
            lines.append("-" * 40)
            lines.append("INSTITUTIONAL OVERLAP (your holdings in tracked funds)")
            lines.append("-" * 40)
            for ticker, holders in overlaps["institutional_overlap"].items():
                lines.append(f"  {ticker}: {', '.join(holders[:3])}")
            lines.append("")

        # Smart money consensus
        if overlaps["smart_money_consensus"]:
            lines.append("-" * 40)
            lines.append("SMART MONEY CONSENSUS (in both Congress + funds)")
            lines.append("-" * 40)
            lines.append(f"  {', '.join(overlaps['smart_money_consensus'][:15])}")
            lines.append("")

        # New ideas
        new = overlaps["new_ideas"]
        if new["in_both_sources"]:
            lines.append("-" * 40)
            lines.append("NEW IDEAS (in tracked sources, not in your portfolio)")
            lines.append("-" * 40)
            lines.append(f"  Both sources: {', '.join(new['in_both_sources'][:10])}")
            if new["congress_only"]:
                lines.append(f"  Congress only: {', '.join(new['congress_only'][:10])}")
            if new["institutional_only"]:
                lines.append(f"  Institutional only: {', '.join(new['institutional_only'][:10])}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
