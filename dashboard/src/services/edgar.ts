import type { FilingSummary } from "@/types";

const EDGAR_SEARCH_BASE = "https://efts.sec.gov/LATEST/search-index";
const EDGAR_SUBMISSIONS_BASE = "https://data.sec.gov/submissions";
const EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index";

const HEADERS = {
  "User-Agent": "FinanceNewsPlatform contact@example.com",
  Accept: "application/json",
};

interface EdgarFiling {
  accessionNumber: string;
  filingDate: string;
  form: string;
  primaryDocument: string;
  primaryDocDescription: string;
}

// Look up CIK from ticker using the SEC company tickers JSON
const tickerToCik = new Map<string, string>();

async function getCik(ticker: string): Promise<string | null> {
  if (tickerToCik.has(ticker)) return tickerToCik.get(ticker)!;

  try {
    const res = await fetch(
      "https://www.sec.gov/files/company_tickers.json",
      { headers: HEADERS, signal: AbortSignal.timeout(10000) }
    );
    if (!res.ok) return null;

    const data = await res.json();
    for (const entry of Object.values(data) as Array<{
      cik_str: number;
      ticker: string;
      title: string;
    }>) {
      tickerToCik.set(
        entry.ticker.toUpperCase(),
        String(entry.cik_str).padStart(10, "0")
      );
    }

    return tickerToCik.get(ticker.toUpperCase()) || null;
  } catch {
    console.error("Failed to fetch CIK mapping");
    return null;
  }
}

export async function fetchFilings(
  ticker: string,
  filingType: string = "10-K"
): Promise<FilingSummary | null> {
  try {
    const cik = await getCik(ticker);
    if (!cik) return null;

    const res = await fetch(
      `${EDGAR_SUBMISSIONS_BASE}/CIK${cik}.json`,
      { headers: HEADERS, signal: AbortSignal.timeout(10000) }
    );
    if (!res.ok) return null;

    const data = await res.json();
    const recent = data.filings?.recent;
    if (!recent) return null;

    // Find the most recent filing of the requested type
    const idx = recent.form?.findIndex(
      (f: string) => f === filingType
    );
    if (idx === -1 || idx === undefined) return null;

    const accession = recent.accessionNumber[idx]?.replace(/-/g, "");
    const filingDate = recent.filingDate[idx];
    const primaryDoc = recent.primaryDocument[idx];

    const filingUrl = `https://www.sec.gov/Archives/edgar/data/${parseInt(cik)}/${accession}/${primaryDoc}`;

    return {
      ticker: ticker.toUpperCase(),
      filingType,
      filingDate,
      url: filingUrl,
      title: `${ticker.toUpperCase()} ${filingType} filed ${filingDate}`,
      summary: `Latest ${filingType} filing for ${ticker.toUpperCase()}, filed on ${filingDate}. View the full document at SEC EDGAR.`,
    };
  } catch {
    console.error(`Failed to fetch EDGAR filing for ${ticker}`);
    return null;
  }
}

export async function searchEdgar(
  query: string,
  forms: string[] = ["10-K", "10-Q", "8-K"]
): Promise<FilingSummary[]> {
  try {
    const formFilter = forms.map((f) => `"${f}"`).join(",");
    const res = await fetch(
      `https://efts.sec.gov/LATEST/search-index?q=${encodeURIComponent(query)}&forms=${formFilter}&dateRange=custom&startdt=${getOneYearAgo()}&enddt=${getToday()}`,
      { headers: HEADERS, signal: AbortSignal.timeout(10000) }
    );

    if (!res.ok) return [];

    const data = await res.json();
    const hits = data.hits?.hits || [];

    return hits.slice(0, 10).map(
      (hit: {
        _source: {
          file_date: string;
          form_type: string;
          display_names: string[];
          file_num: string;
        };
        _id: string;
      }) => ({
        ticker: query.toUpperCase(),
        filingType: hit._source.form_type,
        filingDate: hit._source.file_date,
        url: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum=${hit._source.file_num}`,
        title: `${hit._source.display_names?.[0] || query} - ${hit._source.form_type}`,
        summary: `${hit._source.form_type} filing from ${hit._source.file_date}`,
      })
    );
  } catch {
    console.error(`EDGAR search failed for: ${query}`);
    return [];
  }
}

function getToday(): string {
  return new Date().toISOString().split("T")[0];
}

function getOneYearAgo(): string {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 1);
  return d.toISOString().split("T")[0];
}
