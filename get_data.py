import os
from multiprocessing import Manager, Process
from collections import defaultdict
from itertools import chain
import wrds
import pandas as pd
import pandas_market_calendars as mcal


def get_main_data(db, date, symbols, freq):
    year = date[:4]    
    
    data = db.raw_sql(
    '''
    select
        sym_root as symbol,
        date_trunc('{freq}', min(time_m)) as time,
        max(price) as high,
        min(price) as low,
        sum(size) as volume,
        count(*) as n_transactions
    from taqm_{year}.ctm_{date}
    where sym_root in {symbols}
    group by date_trunc('{freq}', time_m), sym_root
    '''.format(date=date, year=year, symbols=symbols, freq=freq)
    )
    return data


def get_oc_data(db, date, symbols, freq, orient):
    year = date[:4]
    order_map = {'open' : 'asc', 'close' : 'desc'}
    order = order_map[orient]

    data = db.raw_sql(
    '''
    with _{order} as (
        select
            *,
            row_number() over (partition by date_trunc('{freq}', time_m), sym_root order by time_m, tr_seqnum {order})
        from taqm_{year}.ctm_{date}
        where sym_root in {symbols}
        )

    select
        sym_root as symbol,
        date_trunc('{freq}', time_m) as time,
        price as {orient}
    from _{order}
    where row_number = 1
    '''.format(date=date, year=year, symbols=symbols, freq=freq, orient=orient, order=order)
    )

    return data

def get_price_data(db, date, symbols, freq='minute'):
    o = get_oc_data(db, date, symbols, freq, 'open')
    c = get_oc_data(db, date, symbols, freq, 'close')
    main = get_main_data(db, date, symbols, freq)
    data = o.merge(c).merge(main)
    
    return data



# Analyst codes are re-shuffled, what about estimator codes?
# recorded dates are not the dates estimates were made, but when they were entered into the database
# https://wrds-www.wharton.upenn.edu/pages/grid-items/ibes-wrds-101-introduction-and-research-guide/
# https://www.library.kent.edu/files/IBES_GuideUS.pdf


# clean these functions up with holidays...

def get_lower_date_range(date, n):
    lower = date - pd.Timedelta(n, 'D')
    date_range = pd.bdate_range(lower, date)
    date_range = [x for x in date_range if x not in holidays]
    count = 1
    while len(date_range) < n:
        lower = date -  pd.Timedelta(n + count, 'D')
        date_range = pd.bdate_range(lower, date)
        date_range = [x for x in date_range if x not in holidays]
        count += 1
    return date_range


def get_upper_date_range(date, n):
    upper = date + pd.Timedelta(n, 'D')
    date_range = pd.bdate_range(upper, date)
    date_range = [x for x in date_range if x not in holidays]
    count = 1
    while len(date_range) < n:
        upper = date +  pd.Timedelta(n + count, 'D')
        date_range = pd.bdate_range(date, upper)
        date_range = [x for x in date_range if x not in holidays]
        count += 1
    return date_range


def get_date_range(date, n):
    lower_range = get_lower_date_range(date, n)
    upper_range = get_upper_date_range(date, n)
    date_range = set(chain(*[lower_range, upper_range]))
    return date_range

# if we trade the period -7 days, are some of the estimates / actuals included in that
def get_all_dates(symbols, estimates_summary):
    '''
    Returns a dictionary mapping:
        {
        trading_date_i : 
          [(symbol_X, earnings_date_i_X), (symbol_Y, earnings_date_i_Y) ...],
        ...
        trading_date_j : 
          [(symbol_X, earnings_date_j_X), (symbol_Z, earnings_date_j_Z) ...],
        ...
        }
    '''    
    dates_symbols = defaultdict(list)

    for symbol in symbols:
        symbol_dates = estimates_summary.loc[estimates_summary['symbol'] == symbol, 'earnings_date']
        for symbol_date in symbol_dates:
            symbol_date_range = get_date_range(symbol_date, 8) # 8 for +-7 days from earnings date
            for date in symbol_date_range:
                dates_symbols[date].append((symbol, symbol_date))
    
    return dates_symbols


def get_estimates(symbols):

    estimates = db.raw_sql(
        '''
        select
            oftic as symbol,
            anndats_act as earnings_date,
            estimator as firm_code,
            analys as analyst_code,
            fpi as q_out,
            value as estimate,
            actual
        from ibes.det_epsus
        where oftic in {symbols} and anndats_act > '2004-01-08' and fpi in ('6', '7', '8', '9')
        '''.format(symbols=str(symbols))
    )

    estimates['q_out'] = estimates['q_out'].map({'6' : 0, '7' : 1, '8' : 2, '9' : 3})

    return estimates

def summarize_estimates(estimates):
    f = {'estimate' : ['count', 'min', 'max', 'mean', 'median', 'std'], 'actual' : ['mean']}
    estimates_summary = estimates.groupby(['symbol', 'earnings_date']).agg(f)
    estimates_summary.columns = ['n_est', 'min_est', 'max_est', 'mean_est', 'med_est', 'std_est', 'actual']
    estimates_summary = estimates_summary.reset_index()
    estimates_summary['earnings_date'] = estimates_summary['earnings_date'].pipe(pd.to_datetime, format='%Y-%m-%d')
    
    return estimates_summary

def get_daily_data(db, symbols):
    ticker_map = db.raw_sql(
        '''
        select 
            cusip, 
            ticker 
        from crsp.stocknames 
        where ticker in {symbols}
        '''.format(symbols=str(tuple(symbols)))
    ).drop_duplicates()

    cusips = str(tuple(ticker_map['cusip']))

    daily_data = db.raw_sql(
        '''
        select
            date,
            cusip,
            openprc as open,
            bidlo as low,
            askhi as high,
            prc as close,
            vol as volume
        from crsp_a_stock.dsf
        where cusip in {cusips}
        '''.format(cusips=cusips)
    )

    symbol_map = ticker_map.set_index('cusip')['ticker'].to_dict()
    daily_data['symbol'] = daily_data['cusip'].map(symbol_map)
    daily_data['date'] = daily_data['date'].pipe(pd.to_datetime, format='%Y-%m-%d')
    
    return daily_data

# assumes:
# (1) an environment variable, WRDS_USER, is set for the WRDS username 
# (2) a .pgpass file for the WRDS user exists - see:
# https://github.com/wharton/wrds/blob/master/wrds/sql.py

def worker_func(q):
    db = wrds.Connection(wrds_username=os.environ['WRDS_USER'])
    while not q.empty():
        date_str, symbols_str, earnings_date_map = q.get()
        data = get_price_data(db, date_str, symbols_str)
        data['earnings_date'] = data['symbol'].map(earnings_date_map)
        data.to_csv('raw_data/{}.csv'.format(date_str), index=False)

if __name__ == '__main__':
    db = wrds.Connection(wrds_username=os.environ['WRDS_USER'])

    nyse = mcal.get_calendar('NYSE')
    calendar = nyse.schedule(start_date='2003-01-01', end_date='2020-12-31')

    holidays = [
        x for x in pd.date_range('2003-01-01', '2020-12-31') 
        if x not in calendar['market_open'].dt.date
        and x.dayofweek < 5 # represents Monday - Friday
    ]

    symbols = (
        'AMZN', # Amazon
        'COST', #Coscto
        'KR', #Kroger
        'WBA', #Walgreens
        'WMT' #Walmart
    )

    estimates = get_estimates(symbols)
    estimates_summary = summarize_estimates(estimates)
    
    estimates.to_csv('summarized_data/estimates.csv', index=False)
    estimates_summary.to_csv('summarized_data/estimates_summary.csv', index=False)

    daily_data = get_daily_data(db, symbols)
    daily_data.to_csv('summarized_data/daily_data.csv', index=False)
    
    db.close()
    
    dates_info = get_all_dates(symbols, estimates_summary)

    m = Manager()
    q = m.Queue()

    for date, metadata in dates_info.items():
        symbols = tuple(x[0] for x in metadata)
        earnings_dates = tuple(x[1] for x in metadata)
        earnings_dates_map = dict(zip(symbols, earnings_dates))
        date_str = date.strftime('%Y%m%d')
        symbols_str = str(symbols).replace(',)', ')')

        q.put((date_str, symbols_str, earnings_dates_map))

    p_list = []
    
    # 5 is the max rate limit for a student WRDS account
    for _ in range(5):
        p = Process(target=worker_func, args=(q,))
        p_list.append(p)
        p.start()
    for p in p_list:
        p.join()
        p.close()