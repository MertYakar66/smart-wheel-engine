import type { MacroData } from "@/types";

const FRED_BASE = "https://api.stlouisfed.org/fred/series/observations";

// FRED API is free but requires an API key — set FRED_API_KEY env var
// Get one at: https://fred.stlouisfed.org/docs/api/api_key.html
function getApiKey(): string {
  return process.env.FRED_API_KEY || "";
}

const SERIES_DESCRIPTIONS: Record<string, string> = {
  CPIAUCSL: "Consumer Price Index for All Urban Consumers",
  FEDFUNDS: "Federal Funds Effective Rate",
  UNRATE: "Unemployment Rate",
  GDP: "Gross Domestic Product",
  T10Y2Y: "10-Year minus 2-Year Treasury Spread",
  DGS10: "10-Year Treasury Constant Maturity Rate",
  VIXCLS: "CBOE Volatility Index (VIX)",
  SP500: "S&P 500 Index",
  DEXUSEU: "US / Euro Foreign Exchange Rate",
  DCOILWTICO: "Crude Oil Prices: WTI",
};

export async function fetchFredSeries(
  seriesId: string
): Promise<MacroData | null> {
  const apiKey = getApiKey();
  if (!apiKey) {
    console.error("FRED_API_KEY not set — cannot fetch macro data");
    return null;
  }

  try {
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);

    const params = new URLSearchParams({
      series_id: seriesId,
      api_key: apiKey,
      file_type: "json",
      observation_start: oneYearAgo.toISOString().split("T")[0],
      sort_order: "desc",
      limit: "365",
    });

    const res = await fetch(`${FRED_BASE}?${params}`, {
      signal: AbortSignal.timeout(10000),
    });

    if (!res.ok) return null;

    const data = await res.json();
    const observations = data.observations || [];

    return {
      series: seriesId,
      description:
        SERIES_DESCRIPTIONS[seriesId] || seriesId,
      data: observations
        .filter((o: { value: string }) => o.value !== ".")
        .map((o: { date: string; value: string }) => ({
          date: o.date,
          value: parseFloat(o.value),
        }))
        .reverse(),
    };
  } catch {
    console.error(`Failed to fetch FRED series: ${seriesId}`);
    return null;
  }
}

export function getAvailableSeries(): {
  id: string;
  description: string;
}[] {
  return Object.entries(SERIES_DESCRIPTIONS).map(
    ([id, description]) => ({ id, description })
  );
}
