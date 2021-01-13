### Business Use Case

1. *Statement of Problem:* Investment firm would like to develop a trading strategy to profit off differences in earnings reports and analyst estimates for NYSE-listed companies in the retail industry.

2. *Client:* Investment firm

3. *Key Business Question:* Are there common factors in the news and trading leading up to an earnings report, as well as historical patterns, that could generate a profitable trading strategy either prior to or immediately following an earnings report from a retail firm?

4. *Data source(s):*
Preliminary list of retail stock symbols: https://topforeignstocks.com/stock-lists/the-full-list-of-department-store-stocks-trading-on-the-nyse/
Historical analyst estimates (still need access to database): https://www.refinitiv.com/en/financial-data/company-data/institutional-brokers-estimate-system-ibes
SEC filings data: https://www.sec.gov/dera/data/financial-statement-data-sets.html
Macroeconomic data (still need access to database): https://www.refinitiv.com/en/financial-data/market-data/economic-data

5. *Business impact of work:* 
Consider the following 5 retailers listed in descending order of revenue for 2020:  
* Walmart (WMT)
* Amazon (AMZN)
* Costco (COST)
* Walgreens (WBA)
* Kroger (KR)

Now suppose we have \$10,000 in our investment account and we would like to evenly apply this money across the companies to trade the Q3 2020 earnings season.

Within the 7 days following their Q3 2020 earnings reports, the companies saw the following percent deviations in their stock price, rounded to the nearest whole percent:

* Walmart: 4%  
* Amazon: 6%
* Costco: 2%
* Walgreens: -5% 
* Kroger: 3%

-- Source: Yahoo Finance

Suppose we had a perfect trading strategy, where we bought the stocks that increased and shorted those that decreased and took profit at these deviations. We could have earned a profit of:

\begin{equation}
(\$2,000 * 4\%) + ($2,000 * 6\%) + (\$2,000 + 2\%) + (\$2,000 * 5\%) + (\$2,000 * 3\%) = \$2,000
\end{equation}

Or, more simply, a 20% increase in the account value. This is a best case scenario, whereas our more likely profit potential exists below this amount.

6. *How business will use (predicted) model to make decisions(s):* The business will use the model to inform and implement trading strategies during earnings season for this particular market segment.

7. *Metric:* Percent increase in account value in a given year
