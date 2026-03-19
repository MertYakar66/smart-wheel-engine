"""Data schemas for validation and type safety."""

from datetime import date, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class OHLCVSchema(BaseModel):
    """Schema for OHLCV price data."""

    date: date
    ticker: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    vwap: float | None = Field(default=None, gt=0)
    adj_factor: float = Field(default=1.0, gt=0)

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v

    @field_validator("high")
    @classmethod
    def high_gte_open_close(cls, v: float, info) -> float:
        if "open" in info.data and v < info.data["open"]:
            raise ValueError("high must be >= open")
        if "close" in info.data and v < info.data["close"]:
            raise ValueError("high must be >= close")
        return v


class OptionsFlowSchema(BaseModel):
    """Schema for options flow data."""

    date: date
    ticker: str
    call_volume: int = Field(ge=0)
    put_volume: int = Field(ge=0)
    call_oi: int = Field(ge=0)
    put_oi: int = Field(ge=0)
    call_oi_change: int | None = None
    put_oi_change: int | None = None
    put_call_volume_ratio: float | None = Field(default=None, ge=0)
    put_call_oi_ratio: float | None = Field(default=None, ge=0)
    atm_iv: float | None = Field(default=None, ge=0, le=5.0)
    iv_rank: float | None = Field(default=None, ge=0, le=100)
    iv_percentile: float | None = Field(default=None, ge=0, le=100)


class RealizedVolSchema(BaseModel):
    """Schema for realized volatility data."""

    date: date
    ticker: str
    rv_5d: float | None = Field(default=None, ge=0, le=5.0)
    rv_10d: float | None = Field(default=None, ge=0, le=5.0)
    rv_21d: float | None = Field(default=None, ge=0, le=5.0)
    rv_63d: float | None = Field(default=None, ge=0, le=5.0)
    rv_parkinson: float | None = Field(default=None, ge=0, le=5.0)
    rv_garman_klass: float | None = Field(default=None, ge=0, le=5.0)
    rv_yang_zhang: float | None = Field(default=None, ge=0, le=5.0)
    iv_rv_spread: float | None = None


class EarningsSchema(BaseModel):
    """Schema for enriched earnings data."""

    ticker: str
    earnings_date: date
    fiscal_quarter: str
    eps_actual: float | None = None
    eps_estimate: float | None = None
    eps_surprise: float | None = None
    eps_surprise_pct: float | None = None
    revenue_actual: float | None = Field(default=None, ge=0)
    revenue_estimate: float | None = Field(default=None, ge=0)
    revenue_surprise_pct: float | None = None
    guidance_direction: Literal[-1, 0, 1] | None = None
    pre_earnings_iv: float | None = Field(default=None, ge=0, le=5.0)
    post_earnings_iv: float | None = Field(default=None, ge=0, le=5.0)
    earnings_move: float | None = None
    implied_move: float | None = Field(default=None, ge=0)
    move_vs_implied: float | None = None


class FundamentalsSchema(BaseModel):
    """Schema for fundamental data."""

    date: date
    ticker: str
    market_cap: float | None = Field(default=None, ge=0)
    pe_ratio: float | None = None
    pe_forward: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    ev_ebitda: float | None = None
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = Field(default=None, ge=-1, le=1)
    operating_margin: float | None = Field(default=None, ge=-1, le=1)
    net_margin: float | None = Field(default=None, ge=-1, le=1)
    debt_equity: float | None = Field(default=None, ge=0)
    current_ratio: float | None = Field(default=None, ge=0)
    free_cash_flow: float | None = None
    fcf_yield: float | None = None
    dividend_yield: float | None = Field(default=None, ge=0, le=1)
    payout_ratio: float | None = Field(default=None, ge=0)


class NewsSentimentSchema(BaseModel):
    """Schema for news sentiment data."""

    date: date
    ticker: str
    news_count: int = Field(ge=0)
    sentiment_score: float = Field(ge=-1, le=1)
    sentiment_std: float | None = Field(default=None, ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    event_tags: list[str] = Field(default_factory=list)


class FactorExposureSchema(BaseModel):
    """Schema for factor exposure data."""

    date: date
    ticker: str
    factor_value: float | None = None
    factor_growth: float | None = None
    factor_momentum: float | None = None
    factor_quality: float | None = None
    factor_low_vol: float | None = None
    factor_size: float | None = None
    factor_yield: float | None = None


class CorrelationSchema(BaseModel):
    """Schema for correlation data."""

    date: date
    ticker: str
    corr_spx_21d: float | None = Field(default=None, ge=-1, le=1)
    corr_spx_63d: float | None = Field(default=None, ge=-1, le=1)
    corr_sector_21d: float | None = Field(default=None, ge=-1, le=1)
    beta_21d: float | None = None
    beta_63d: float | None = None
    idio_vol: float | None = Field(default=None, ge=0)


class BorrowRateSchema(BaseModel):
    """Schema for borrow rate / short interest data."""

    date: date
    ticker: str
    short_interest: int | None = Field(default=None, ge=0)
    short_interest_ratio: float | None = Field(default=None, ge=0)
    utilization: float | None = Field(default=None, ge=0, le=1)
    borrow_rate: float | None = Field(default=None, ge=0)
    available_shares: int | None = Field(default=None, ge=0)


class MacroEventSchema(BaseModel):
    """Schema for macro events calendar."""

    date: date
    event_type: str
    event_time: str | None = None
    actual: float | None = None
    consensus: float | None = None
    prior: float | None = None
    surprise: float | None = None
    market_impact: float | None = None


class ETFFlowSchema(BaseModel):
    """Schema for ETF flow data."""

    date: date
    etf_ticker: str
    flow_usd: float | None = None
    flow_shares: int | None = None
    aum: float | None = Field(default=None, ge=0)
    flow_pct: float | None = None
    cumulative_flow_30d: float | None = None


class IndexMembershipSchema(BaseModel):
    """Schema for index membership history."""

    date: date
    ticker: str
    action: Literal["ADD", "REMOVE"]
    replacing: str | None = None
