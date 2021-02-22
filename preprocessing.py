import os
import json
from itertools import zip_longest
from itertools import chain

import pandas as pd

files = [file for file in os.listdir('raw_data') if '.csv' in file]

df = []

for file in files:
    tmp = pd.read_csv('raw_data/{}'.format(file))
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
with open('summarized_data/dataset_dtypes.json', 'w') as f:
    json.dump(dtypes, f)

df.to_csv('summarized_data/dataset.csv', index=False)

df = df.set_index(['symbol', 'earnings_date', 'date', 'time'])
df = df.sort_index(ascending=True)

# Filter out hours outside of market hours
idx = pd.IndexSlice
df = df.loc[idx[
    :,
    '2010-01-01':'2019-12-31',
    :,
    '1900-01-01 9:30:00':'1900-01-01 15:59:00'
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
df = df.loc[df['relative_offset'].between(-1,1)]
relative_offset = df['relative_offset']
df = df.drop(['earnings_date_offset', 'relative_offset'], axis=1)

symbols = df.index.get_level_values('symbol').unique()
symbol_dates = {symbol : df.loc[
    (symbol)].index.get_level_values('earnings_date').drop_duplicates()
                for symbol in symbols}

daily_data = pd.read_csv('summarized_data/daily_data.csv')
daily_data = daily_data.drop('cusip', axis=1)
daily_data['date'] = daily_data['date'].pipe(
    pd.to_datetime, format='%Y-%m-%d')
daily_data = daily_data.set_index(['symbol', 'date'])
daily_data = daily_data.sort_index(ascending=True)

estimates = pd.read_csv('summarized_data/estimates.csv')
estimates_summary = pd.read_csv('summarized_data/estimates_summary.csv')
estimates_summary['earnings_date'] = estimates_summary['earnings_date'].pipe(
    pd.to_datetime, format='%Y-%m-%d')
estimates_summary = estimates_summary.set_index(['symbol', 'earnings_date'])

def get_daily_mean(group, n):
    mean = group.rolling(n, min_periods=1).mean()
    return mean

def get_daily_std(group, n):
    std = group.rolling(n, min_periods=1).std()
    std = std.fillna(method='bfill')
    return std

def normalize(group):
    """
    Normalize the estimates data by subtracting the running mean
    and standard deviation.
        
    Parameters
    ---
    group : a Pandas DataFrame containing the data for the group
    """
    
    mean = group.rolling(len(group),min_periods=1).mean()
    std = group.rolling(len(df),min_periods=1).std()
    std = std.fillna(method='bfill')
    normalized_group = (group - mean) / std
    normalized_group.fillna(method='bfill')
    return normalized_group

# get daily 10-day mean and 10-day std to use to normalize
# the price data
daily_mean = daily_data.groupby('symbol').apply(get_daily_mean, 10)
daily_std = daily_data.groupby('symbol').apply(get_daily_std, 10)

mean_price = df[['open', 'high', 'low', 'close']].sum(axis=1) / 4
#normalized_df = df.groupby('symbol').apply(normalize)
daily_cols = ['open', 'high', 'low', 'close', 'volume']

reindexed_df = df.set_index(df.index.droplevel(['earnings_date', 'time']))

daily_mean = daily_mean.loc[reindexed_df.index.drop_duplicates()]
daily_std = daily_std.loc[reindexed_df.index.drop_duplicates()]

normalized_df = reindexed_df[daily_cols].subtract(
    daily_mean[daily_cols], axis=0).divide(
    daily_std[daily_cols], axis=0)

normalized_df.index = df.index
normalized_df['n_transactions'] = df.groupby(
    'symbol')['n_transactions'].apply(normalize)
# add back the relative offset from earnings date
normalized_df['relative_offset'] = relative_offset
df['relative_offset'] = relative_offset


normalized_estimates = estimates_summary.groupby(
    'symbol').apply(normalize)


# split 80/20 sequentially for each symbol
split_factor = .8
symbol_split = {
    symbol : {
        'train' : dates[:int(len(dates) * split_factor)],
        'test' : dates[-(len(dates) - int(len(dates) * split_factor)):]
    } for symbol, dates in symbol_dates.items()
}

# save training / testing data to csv for further analysis
training_ix = [list(zip_longest([], ixes['train'], fillvalue=symbol))
               for symbol, ixes in symbol_split.items()]
training_ix = list(chain.from_iterable(training_ix))
training_df = pd.concat([df.loc[ix] for ix in training_ix], 
                        keys=training_ix, names=['symbol', 'earnings_date'])
training_df.to_csv('summarized_data/train_dataset.csv')

testing_ix = [list(zip_longest([], ixes['test'], fillvalue=symbol)) 
              for symbol, ixes in symbol_split.items()]
testing_ix = list(chain.from_iterable(testing_ix))
testing_df = pd.concat([df.loc[ix] for ix in testing_ix], 
                       keys=training_ix, names=['symbol', 'earnings_date'])
testing_df.to_csv('summarized_data/test_dataset.csv')

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
        A pandas DataFrame of the true average price for each timestep in date_subset
    
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
    
for symbol, dates in symbol_split.items():
    data_subset = normalized_df.loc[(symbol)]
    price_subset = mean_price.loc[(symbol)]
    estimate_subset = normalized_estimates.loc[(symbol)]

    for date in dates['train']:
        write_obs(symbol, date, data_subset, price_subset, estimate_subset, 'processed_data/train')

    for date in dates['test']:
        write_obs(symbol, date, data_subset, price_subset, estimate_subset, 'processed_data/test')