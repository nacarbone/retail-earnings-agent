import os
import json
from itertools import zip_longest
from itertools import chain

import pandas as pd
        
def read_intraday_data():
    """
    Reads each csv file containing the intraday OHLCV data saved to the 
    raw_data directory, combining them  into a single dataset.
    
    Returns
    ---
    A Pandas DataFrame containing the OHLCV data; indexed by
        symbol, earnings date, date and time
        Note: the time field will be formatted as a Pandas Timestamp with date
        1900-01-01
    """    
    filenames = [filename for filename in os.listdir('raw_data') 
                 if '.csv' in file]

    df = []
    for file in files:
        filepath = os.path.join('raw_data', file)
        tmp = pd.read_csv(.format(file))
        tmp['date'] = file.replace('.csv', '')
        df.append(tmp)
    df = pd.concat(df)
    
    df['date'] = df['date'].pipe(
        pd.to_datetime, format='%Y%m%d')
    df['time'] = df['time'].pipe(
        pd.to_datetime, format='0 days %H:%M:%S')
    df['earnings_date'] = df['earnings_date'].pipe(
        pd.to_datetime, format='%Y-%m-%d')
    # save dtypes for easier reading of file
    dtypes = df.dtypes
    dtypes = dtypes.astype(str).to_dict()
    dtypes_filepath = os.path.join('summarized_data', 'dataset_dtypes.json')
    with open(dtypes_filepath, 'w') as f:
        json.dump(dtypes, f)

    df.to_csv('summarized_data/dataset.csv', index=False)

    df = df.set_index(['symbol', 'earnings_date', 'date', 'time'])
    df = df.sort_index(ascending=True)

    return df

def subset_intraday_data(df, data_slice, time_slice, days_out):
    """
    Subsets the OHLCV data based on the values in date_slice, time_slice and
    days_out
    
    Parameters
    ---
    df : a Pandas DataFrame containing the OHCLV data
    date_slice : a slice object for the earnings dates to include
        (e.g. slice('2019-01-01', '2019-12-31') for 2019 only)
    time_slice : a slice object for the intraday time to include
        (e.g. slice('1900-01-01 9:30:00', '1900-01-01 15:59:00') for market 
        hours only)
    days_out : a positive integer for the number of days to/from the earnings
        date to include
        Note: the day immediately following the earnings report as day 0, so
        if earnings were announced on Wednesday after market close, Wednesday
        would be day -1, Thursday day 0 and Friday day 1
    
    Returns
    ---
    (1) A Pandas DataFrame with the subsetted OHLCV values
    (2) The relative trading days to and from the earnings date, to be
        included later for training the model
    """
    
    # Filter out hours outside of market hours
    # and earnings dates prior to 2010 
    idx = pd.IndexSlice
    df = df.loc[idx[
        :,
        date_slice,
        :,
        time_slice
    ]]

    # Add a relative offset of trading periods until earnings
    # Filter this for day before, day of and day after
    df['earnings_date_offset'] = df.index.get_level_values('date') \
        - df.index.get_level_values('earnings_date')
    df['earnings_date_offset'] = df['earnings_date_offset'].dt.days
    relative_offset = df.groupby(['symbol', 'earnings_date']).apply(
        lambda x: x['earnings_date_offset'].drop_duplicates().rank()
    ).droplevel('time').droplevel([2,3])
    relative_offset = relative_offset - 8
    relative_offset.name = 'relative_offset'
    df = df.join(relative_offset)
    # select relative offsets -1, 0, 1, drop these so they aren't 
    # normalized but store to add back later
    df = df.loc[df['relative_offset'].between(-days_out,days_out)]
    relative_offset = df['relative_offset']
    df = df.drop(['earnings_date_offset', 'relative_offset'], axis=1)
    
    return df, relative_offset

def read_daily_data():
    """
    Reads the csv file containining the daily OHLCV data saved to the 
    summarized_data directory
    
    Returns
    ---
    A Pandas DataFrame containing the OHLCV data; indexed by
        symbol and date; to be used for standardization of the
        intraday data
    """
    daily_data = pd.read_csv('summarized_data/daily_data.csv')
    daily_data = daily_data.drop('cusip', axis=1)
    daily_data['date'] = daily_data['date'].pipe(
        pd.to_datetime, format='%Y-%m-%d')
    daily_data = daily_data.set_index(['symbol', 'date'])
    daily_data = daily_data.sort_index(ascending=True)

    return daily_data

def read_estimates():
    """
    Reads the csv file containining the analyst estimates data saved to the 
    summarized_data directory
    
    Returns
    ---
    A Pandas DataFrame containing the analyst estimates; indexed by
        symbol and earnings date
    """    
    estimates_summary = pd.read_csv('summarized_data/estimates_summary.csv')
    estimates_summary['earnings_date'] = estimates_summary['earnings_date'].pipe(
        pd.to_datetime, format='%Y-%m-%d')
    estimates_summary = estimates_summary.set_index(['symbol', 'earnings_date'])
    estimates_summary = estimates_summary.sort_index(ascending=True)
    
    return estimates_summary

def get_split(df, split_factor):
    """
    Aggregates indexes for splitting the data, using a sequential split based
        on the value 
    
    Parameters
    ---
    df : a Pandas DataFrame indexed by at least symbol, earnings date;
        should be the intraday OHLCV data
    split_factor : a float in [0, 1] corresponding to the percentage of 
        symbol-earnings date pairs that will be assigned to the training set
        
    Returns
    ---
    A nested Python dict mapping 
        {
        symbol : 
            {
            'train' : [earnings_date_i1, earnings_date_i2, ...],
            'test' : [earnings_date_j1, earnings_date_j2, ...]
            }
        }
    """
    symbols = df.index.get_level_values('symbol').unique()
    symbol_dates = {symbol : df.loc[
        (symbol)].index.get_level_values('earnings_date').drop_duplicates()
                    for symbol in symbols}
    symbol_split = {
        symbol : {
            'train' : dates[:int(len(dates) * split_factor)],
            'test' : dates[-(len(dates) - int(len(dates) * split_factor)):]
        } for symbol, dates in symbol_dates.items()
    }

    return symbol_split

def save_split_to_file(df, symbol_split, subset_alias):
    """
    Indexes df based on training and testing dates, saving the resulting
    subset to a file in the summarized_data directory for further analysis
    
    Parameters
    ---
    df : a Pandas DataFrame containing the intraday OHLCV values
    symbol_split : the nested Python dict that maps symbol-earnings date
        pairs to either the training or testing set
    subset_alias : 'train' or 'test'
    """
    ix_for_alias = [list(zip_longest([], ixes[subset_alias], 
                                     fillvalue=symbol))
                    for symbol, ixes in symbol_split.items()]
    ix_for_alias = list(chain.from_iterable(ix_for_alias))
    df_for_alias = pd.concat(
        [df.loc[ix] for ix in training_ix],
        keys=training_ix, names=['symbol', 'earnings_date'])
    filename_for_alias = '{}_dataset.csv'.format(subset_alias)
    path_for_alias = os.path.join('summarized_data', filename_for_alias)
    df_for_alias.to_csv(path_for_alias)

def get_daily_mean(group, n):
    """
    Compute the n-length running mean for each symbol's daily OHLCV data
        
        
    Parameters
    ---
    group : a Pandas DataFrame containing the daily OHLCV data for the symbol
    n : the number of days over which to calculate

    Returns
    ---
    A Pandas DataFrame containing the running means for the groups
    """
    return group.rolling(n, min_periods=1).mean()

def get_daily_std(group, n):
    """
    Compute the n-length running standard deviation for each symbol's daily 
        OHLCV data. Fills null (i.e. for the first observation) backward; the
         daily data should start well before the intraday data, making this
         a non-issue
        
    Parameters
    ---
    group : a Pandas DataFrame containing the daily OHLCV data for the symbol
    n : the number of days over which to calculate
    
    Returns
    ---
    A Pandas DataFrame containing the running standard deviations for the 
        groups
    """
    std = group.rolling(n, min_periods=1).std()
    return std.fillna(method='bfill')

def normalize(group):
    """
    A more generalized normalization computed by subtracting the running mean
    and standard deviation. Primarily used for the estimates data
        
    Parameters
    ---
    group : a Pandas DataFrame containing the data for the symbol
    
    Returns
    ---
    A Pandas DataFrame containing the normalized data for the symbol
    """
    
    mean = group.rolling(len(group),min_periods=1).mean()
    std = group.rolling(len(df),min_periods=1).std()
    std = std.fillna(method='bfill')
    normalized_group = (group - mean) / std
    normalized_group.fillna(method='bfill')
    return normalized_group


# get a mean of the price, this won't be normalized, but
# but will be used in 
mean_price = df[['open', 'high', 'low', 'close']].sum(axis=1) / 4

def normalize_intraday_data(df, daily_data, n):
    """
    Normalizes the intraday OHLCV data using the running means and standard
        deviations of the daily OHLCV data
        
    Parameters
    ---
    df : the intraday OHLCV data
    daily_data : the daily OHLCV data
    n : the number of days over which to calculate running means and
        standard deviations from the daily data
    """
    
    # get daily 10-day mean and 10-day std to use to normalize
    # the price data
    reindexed_df = df.set_index(df.index.droplevel(['earnings_date', 'time']))
    daily_mean = daily_data.groupby('symbol').apply(get_daily_mean, 10)
    daily_mean = daily_mean.loc[reindexed_df.index.drop_duplicates()]
    daily_std = daily_data.groupby('symbol').apply(get_daily_std, 10)
    daily_std = daily_std.loc[reindexed_df.index.drop_duplicates()]
    daily_cols = ['open', 'high', 'low', 'close', 'volume']
    normalized_df = reindexed_df[daily_cols].subtract(
        daily_mean[daily_cols], axis=0).divide(
        daily_std[daily_cols], axis=0)
    normalized_df.index = df.index
    normalized_df['n_transactions'] = df.groupby(
        'symbol')['n_transactions'].apply(normalize)
    
    return normalized_df

def write_obs(
    symbol, 
    date, 
    data_subset, 
    price_subset, 
    esimate_subset, 
    output_dir
):
    """
    Write each episode to its own file as a dict of normalized
    price/volume inputs, normalized estimates and the true average price.
    
    Parameters
    ---
    symbol : str
        The symbol for this particular observation
    
    date : pandas.Timestamp
        The earnings date for this particular observation
    
    date_subset : pandas.DataFrame
        A pandas DataFrame of normalized inputs
    
    price_subset : pandas.Series
        A pandas DataFrame of the true average price for each timestep in 
            date_subset
    
    estimate_subset: pandas.DataFrame
        A pandas DataFrame of the normalized estimates
    
    output_dir : str
        The location where the files will be written
    """

    data_date_subset = data_subset.loc[(date)].values.tolist()
    price_date_subset = price_subset.loc[(date)].values.tolist()
    estimate_date_subset = estimate_subset.loc[(date)].values.tolist()

    date_str = date.strftime('%Y%m%d')
    
    episode = {'data' : data_date_subset, 'price' : price_date_subset, 'estimate' : estimate_date_subset}
    
    output_file = '{output_dir}/{symbol}-{date}.json'.format(output_dir=output_dir, symbol=symbol, date=date_str)
    
    with open(output_file, 'w') as f:
        json.dump(episode, f)

DATE_SLICE = slice('2010-01-01','2019-12-31')
TIME_SLICE = slice('1900-01-01 9:30:00','1900-01-01 15:59:00')
DAYS_OUT = 1
SPLIT_FACTOR = .8
N = 10
TRAIN_DIR = os.path.join('processed_data', 'train')
TEST_DIR = os.path.join('processed_data', 'test')

df = read_intraday_data()
df, relative_offset = subset_intraday_data(
    df, 
    DATE_SLICE, 
    TIME_SLICE, 
    DAYS_OUT
)
estimates_summary = read_estimates()
daily_data = read_daily_data()

normalized_df = normalize_intraday_data(df, daily_data, N)
normalized_estimates = estimates_summary.groupby(
    'symbol').apply(normalize)

# add back the relative offset
normalized_df['relative_offset'] = relative_offset
df['relative_offset'] = relative_offset

symbol_split = get_split(df, SPLIT_FACTOR)
save_split_to_file(df, symbol_split, 'train')
save_split_to_file(df, symbol_split, 'test')

for symbol, dates in symbol_split.items():
    data_subset = normalized_df.loc[(symbol)]
    price_subset = mean_price.loc[(symbol)]
    estimate_subset = normalized_estimates.loc[(symbol)]

    for date in dates['train']:
        write_obs(symbol, date, data_subset, price_subset, estimate_subset, TRAIN_DIR)

    for date in dates['test']:
        write_obs(symbol, date, data_subset, price_subset, estimate_subset, TEST_DIR)