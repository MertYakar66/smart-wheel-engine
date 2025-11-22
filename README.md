# smart-wheel-engine
1. Executive Summary
   
This project develops a systematic, data-driven framework for managing short-put and covered-call strategies. Instead of relying on fixed rules or intuition, the system evaluates each potential trade using probability estimates and expected-value calculations derived from historical data.
My goal is to only enter trades that offer positive statistical expectancy and avoid trades where the risk is not justified by the potential premium.
To achieve this, the project uses historical price data, option chain information, and volatility measures to build a dataset of past trade outcomes. Machine-learning models then learn how different market conditions affect the probability and profitability of short-put and covered-call trades. Finally, a decision engine selects the best opportunities while applying strict risk controls and market-regime filters.
This produces a structured, rational, and verifiable process for managing the Wheel Strategy and helps identify the trades only when the historical data supports them.

