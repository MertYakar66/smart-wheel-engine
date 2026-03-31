import type {
  DataProvider,
  Headline,
  Quote,
  FilingSummary,
  MacroData,
} from "@/types";
import { ingestAllFeeds } from "./rss-ingestion";
import {
  fetchQuoteFromFinnhub,
  getLatestSnapshot,
} from "./market-data";
import { fetchFilings } from "./edgar";
import { fetchFredSeries } from "./macro-data";

class FinanceDataProvider implements DataProvider {
  async fetchHeadlines(): Promise<Headline[]> {
    return ingestAllFeeds();
  }

  async fetchQuote(ticker: string): Promise<Quote | null> {
    // Try live first, fall back to cached
    const live = await fetchQuoteFromFinnhub(ticker);
    if (live) return live;
    return getLatestSnapshot(ticker);
  }

  async fetchFilingSummary(
    ticker: string,
    filingType: string
  ): Promise<FilingSummary | null> {
    return fetchFilings(ticker, filingType);
  }

  async fetchMacroData(series: string): Promise<MacroData | null> {
    return fetchFredSeries(series);
  }
}

// Singleton
export const dataProvider = new FinanceDataProvider();
