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

(\$2,000 * 4\%) + ($2,000 * 6\%) + (\$2,000 + 2\%) + (\$2,000 * 5\%) + (\$2,000 * 3\%) = \$2,000

Or, more simply, a 20% increase in the account value. This is a best case scenario, whereas our more likely profit potential exists below this amount.

Early results of the the model suggest a 2% improvement over a baseline model of random decisionmaking. This is expected to increase as the model is further trained and tuned.

6. *How business will use (predicted) model to make decisions(s):* The business will use the model to inform and implement trading strategies during earnings season for this particular market segment.

7. *Metric:* Percent increase in account value in a given year

8. *Methodology:*

# Environment

The environment and its states consists of four components:
<ol>
    <li>The one-minute values of for open, high, low, close and volume for the stock's shares</li>
    <li>Analyst estimates and their summary statistics for this earnings date</li>
    <li>The cash balance and number of shares held in the account
    <li>The acceptable list of actions based on the accounts' position (e.g. an account with 50 shares cannot sell >50 shares</li>
</ol>
 
# Model Design

The baseline and RL models, respectively, are designed as follows: 
<ol>
    <li>At each timestep, the baseline model randomly selects an action to buy, sell or hold its current position. Based on this choice, it randomly selects an amount of shares to buy or sell based on its current cash balance and number of shares (i.e. the model can only sell as many shares as it actually has in its account. This process is similar to a random walk and would represent an investor whose trading decisions are completely arbitrary. This process is repeated over 100 episodes and the results averaged.</li>
    <li>At each timestep, the RL model is faced with the same set of decisions as the baseline model; however, it approaches its decisionmaking much more intelligently. Instead of guessing randomly, the model probabilistically chooses to explore new actions or exploit its own knowledge of the underlying trading process. At each timestep, a neural network outputs an encoding of the current state to be fed into an action distribution. The action distribution consists of another neural network which produces the logits of its action type (buy/sell/hold) as well as the embedding of the amount to buy/sell/hold, conditioned on the action type (thus making the distribution autoregressive). The embedding is then converted to logits using a fixed embedding matrix given by the environment. Any unavailble actions are masked out of the logits. The model selects actions from this distribution according to a policy, which is optimized according to a Proximal Policy Optimization (PPO) algorithm. The policy loss backpropagates through the networks after sequences of 50 timesteps. Due to computational limits of the computer that the model is currently running on, the model has only been run for approximately two half-episodes. The architectures of the neural networks are listed in the appendix.
</ol>

9. *Model Architecture:*

![alt text](assets/architecture.png)