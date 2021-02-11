import os
import json
import pandas as pd

files = [file for file in os.listdir('raw_data') if '.csv' in file]

df = []

for file in files:
    tmp = pd.read_csv('raw_data/{}'.format(file))
    tmp['date'] = file.replace('.csv', '')
    df.append(tmp)

df = pd.concat(df)

# originally the index was saved with the source files, the script has been edited to avoid this

if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

estimates = pd.read_csv('summarized_data/estimates.csv')
estimates_summary = pd.read_csv('summarized_data/estimates_summary.csv')

df['date'] = df['date'].pipe(pd.to_datetime, format='%Y%m%d')
df['time'] = df['time'].pipe(pd.to_datetime, format='0 days %H:%M:%S')
df['earnings_date'] = df['earnings_date'].pipe(pd.to_datetime, format='%Y-%m-%d')
estimates_summary['earnings_date'] = estimates_summary['earnings_date'].pipe(pd.to_datetime, format='%Y-%m-%d')

# save dtypes for easier reading of file
dtypes = df.dtypes
dtypes = dtypes.astype(str).to_dict()
with open('summarized_data/dataset_dtypes.json', 'w') as f:
    json.dump(dtypes, f)

df.to_csv('summarized_data/dataset.csv', index=False)

df = df.set_index(['symbol', 'earnings_date', 'date', 'time'])
estimates_summary = estimates_summary.set_index(['symbol', 'earnings_date'])

df.sort_index(ascending=True, inplace=True)

# Filter out hours outside of market hours
idx = pd.IndexSlice
df = df.loc[idx[:,:,:,'1900-01-01 9:30:00':'1900-01-01 15:59:00']]

symbols = df.index.get_level_values('symbol').unique()
symbol_dates = {symbol : df.loc[(symbol)].index.get_level_values('earnings_date').drop_duplicates() 
                         for symbol in symbols}


def normalize(group):
    """
    Normalize the data by subtracting the running mean and standard deviation.
    Because there are gaps in our time period, this isn't a perfect approach (e.g. a stock price
    naturally inflates over time so 3 months from now it might be much larger than its current mean).
    
    TODO: Find better approach
    A better approach might be to use the data we're not looking at to help normalize the observations
    (i.e. calculate mean and std even during periods we're not looking at)
    
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

mean_price = df[['open', 'high', 'low', 'close']].sum(axis=1) / 4
normalized_df = df.groupby('symbol').apply(normalize)
# some early periods are still null due to 0 standard deviation, so just fill these with 0
# becuase ths is due to repeated values in the beginning of the period
normalized_df = normalized_df.fillna(0)
normalized_estimates = estimates_summary.groupby('symbol').apply(normalize)

# split sequentially for each group

split_factor = .8
symbol_split = {
    symbol : {
        'train' : dates[:int(len(dates) * split_factor)],
        'test' : dates[-(len(dates) - int(len(dates) * split_factor)):]
    } for symbol, dates in symbol_dates.items()
}

def write_obs(symbol, date, data_subset, price_subset, esimate_subset, output_dir):
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

