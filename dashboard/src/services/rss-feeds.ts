import type { RSSFeedConfig } from "@/types";

export const RSS_FEEDS: RSSFeedConfig[] = [
  {
    name: "Reuters Business",
    url: "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    publisher: "Reuters",
  },
  {
    name: "CNBC Top News",
    url: "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    publisher: "CNBC",
  },
  {
    name: "MarketWatch Top Stories",
    url: "https://feeds.marketwatch.com/marketwatch/topstories/",
    publisher: "MarketWatch",
  },
  {
    name: "WSJ Markets",
    url: "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    publisher: "WSJ",
  },
  {
    name: "Yahoo Finance",
    url: "https://finance.yahoo.com/news/rssindex",
    publisher: "Yahoo Finance",
  },
  {
    name: "Seeking Alpha Market News",
    url: "https://seekingalpha.com/market_currents.xml",
    publisher: "Seeking Alpha",
  },
  {
    name: "Financial Times",
    url: "https://www.ft.com/rss/home",
    publisher: "Financial Times",
  },
  {
    name: "Bloomberg Markets",
    url: "https://feeds.bloomberg.com/markets/news.rss",
    publisher: "Bloomberg",
  },
];
