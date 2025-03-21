import pandas as pd
import itertools


from Quandl import *
import Quandl.Quandl_Python_Tables_API as quandl_api



def calc_ratios_report(df):
    merged = df.copy()
    merged['filing_date'] = pd.to_datetime(merged['filing_date'])
    merged['date'] = pd.to_datetime(merged['date'])
    merged['per_end_date'] = pd.to_datetime(merged['per_end_date'])
    merged['debt'] = merged['net_lterm_debt'].fillna(merged['tot_lterm_debt'])
    merged['EPS'] = merged['eps_diluted_net'].fillna(merged['basic_net_eps'])
    merged['EPS'] = merged['EPS'].apply(lambda x: max(x, 0.001))
    merged['R'] = merged['ret_invst'] * (merged['debt'] +merged['mkt_val'])
    merged = batch_process(merged)
    merged['price_ratio'] = merged['adj_close'] / merged['quarter_prices']
    merged['daily_mkt_val'] = merged['mkt_val'] * merged['price_ratio']
    merged['daily D/M'] = (merged['tot_debt_tot_equity'] * merged['mkt_val']) / merged['daily_mkt_val']
    merged['daily_roi'] = merged['R'] / (merged['debt'] + merged['daily_mkt_val'])
    merged['daily P/E'] = merged['adj_close'] / merged['EPS']
#
    merged = merged[[
    'ticker',
    'adj_close',
    'date',
    'per_end_date',
    'filing_date',
    'effective_date',
    'quarter_prices',
    'price_ratio',
    'daily_roi',
    'daily P/E',
    'daily D/M',
    'R',
    'EPS',
    'ret_invst',
    'mkt_val',
    'debt',
    'net_lterm_debt',
    'tot_lterm_debt',
    'eps_diluted_net',
    'basic_net_eps',
    'shares_out',
    'tot_debt_tot_equity'

]]

    return merged


def process_ticker(group):
    """Efficiently calculate quarter-end prices for each ticker."""
    # Work with the subset of columns required for calculations
    per_end_price = group.copy()

    # Initialize 'quarter_price_2' column
    per_end_price['quarter_price'] = None

    # Get unique period-end dates
    unique_dates = per_end_price['per_end_date'].dropna().unique()

    # Select prices for unique period-end dates
    prices_at_dates = per_end_price.loc[per_end_price['date'].isin(unique_dates), ['date', 'adj_close']]
    prices_at_dates.rename(columns={'date': 'per_end_date'}, inplace=True)

    # Merge prices back into the original DataFrame
    group_test = per_end_price.merge(prices_at_dates, on='per_end_date', how='left')

    # Forward-fill missing values for quarter-end prices
    group_test['quarter_prices'] = group_test['adj_close_y'].ffill()

    # Return only the required columns
    return group_test



def batch_process(df, batch_size=50):
    """Process tickers in batches for efficiency."""
    unique_tickers = df['ticker'].unique()
    results = []
    for i in range(0, len(unique_tickers), batch_size):
        batch_tickers = unique_tickers[i:i + batch_size]
        batch_df = df[df['ticker'].isin(batch_tickers)]

        # Apply processing logic for each batch
        processed_batch = batch_df.groupby('ticker', group_keys=False).apply(process_ticker)
        results.append(processed_batch)

    final = pd.concat(results, ignore_index=True)
    final.drop(columns='adj_close_y', inplace=True)
    final.rename(columns={'adj_close_x': 'adj_close'}, inplace=True)

    return final





def load_data(quandl_api):
    """
    Loads data from Quandl tables, applies necessary filtering, and merges them into a final DataFrame.
    """

    table_names = ['ZACKS/FC', 'ZACKS/FR', 'ZACKS/MT', 'ZACKS/MKTV', 'ZACKS/SHRS']

    # Fetch tables and store them in a dictionary
    data_dict = {
        name.replace('/', '_'): quandl_api.fetch_quandl_table(name, avoid_download=True)
        for name in table_names
    }

    # Extract individual DataFrames
    mktv = data_dict['ZACKS_MKTV'][['ticker', 'per_end_date', 'mkt_val']]
    fr = data_dict['ZACKS_FR'][['ticker', 'per_end_date', 'tot_debt_tot_equity', 'ret_invst']]
    fc = data_dict['ZACKS_FC'][['ticker', 'per_end_date', 'net_lterm_debt', 'tot_lterm_debt',
                                'eps_diluted_net', 'basic_net_eps', 'filing_date']]
    shrs = data_dict['ZACKS_SHRS'][['ticker', 'per_end_date', 'shares_out']]
    mt = data_dict['ZACKS_MT'][['ticker', 'zacks_x_sector_code']]

    # Apply filters
    date_filter = '2017-01-01'
    mktv = mktv[(mktv['per_end_date'] > date_filter) & (mktv['mkt_val'] * 1000 > 1e6)]
    fr = fr[(fr['per_end_date'] > date_filter) & (fr['tot_debt_tot_equity'] > 0.1)]
    fc = fc[fc['per_end_date'] > date_filter]
    shrs = shrs[shrs['per_end_date'] > date_filter]

    # Exclude sectors
    exclude_numbers = {5, 13}
    mt = mt[~mt['zacks_x_sector_code'].isin(exclude_numbers)][['ticker']]

    # Merge DataFrames
    df = (
        fc.merge(mktv, on=['ticker', 'per_end_date'], how='inner')
        .merge(fr, on=['ticker', 'per_end_date'], how='inner')
        .merge(shrs, on=['ticker', 'per_end_date'], how='inner')
        .merge(mt, on=['ticker'], how='inner')
        .dropna()
    )

    df['per_end_date'] = pd.to_datetime(df['per_end_date'])
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['effective_date'] = df['filing_date'] + pd.Timedelta(days=1)
    df['effective_date'] = pd.to_datetime(df['effective_date'])

    return df

import random


def valid_tickers_and_sample(df):


    """
    Ensures tickers report through full sample
    Creates a sample of 250 tickers
    """


    df_max_dates = df.groupby(['ticker'])['per_end_date'].max()
    valid_tickers = df_max_dates[df_max_dates >= '2024-06-01'].index
    df = df[df['ticker'].isin(valid_tickers)]

    ticker_counts = df.groupby('ticker')['per_end_date'].count()
    tickers_gt_28 = ticker_counts[ticker_counts > 20]
    tickers_list = list(tickers_gt_28.index)
    df = df[df['ticker'].isin(tickers_list)]

    tickers = df.ticker.unique()
    tickers = list(set(tickers))

    random.seed(100)
    sample_tickers = random.sample(tickers, 250)

    sample_df = df[df['ticker'].isin(sample_tickers)]
    sample_df = sample_df.sort_values(['ticker', 'effective_date'])

    return sample_df, sample_tickers


def stock_prices(sample_tickers, download = False):

    """
    Loads data from QUANDL (saved as a CSV for convenience)
    Filters columns and adjusts dates
    Ensures only sample tickers from the fundamental data are being used
    """
    if download:
       stock_prices = quandl_api.fetch_quandl_table("QUOTE/MEDIA", avoid_download=True)
       stock_prices = stock_prices[stock_prices['date'] > '2017-01-01']
       stock_prices = stock_prices[['ticker', 'adj_close', 'date']]
       stock_prices_sample = stock_prices[stock_prices['ticker'].isin(sample_tickers)]
       price_sample = stock_prices_sample

    else:


        stock_prices = pd.read_csv("stock_prices.csv")
        stock_prices = stock_prices[['ticker', 'adj_close', 'date']]
        stock_prices_sample = stock_prices[stock_prices['ticker'].isin(sample_tickers)]
        price_sample = stock_prices_sample

    return price_sample


def merge_samples(price_sample, fund_sample):

    """
    Merges fundamental Zacks data with daily price data
    Accounts for date adjustments

    """


    fund_sample = fund_sample.drop_duplicates(
        subset=['ticker', 'effective_date'],
        keep='last'
    ).sort_values(['ticker', 'effective_date'])
    fund_sample = fund_sample[~fund_sample['ticker'].isin(['BIO.B', 'MOG.B'])]

    price_sample['date'] = pd.to_datetime(price_sample['date'])
    fund_sample['effective_date'] = pd.to_datetime(fund_sample['effective_date'])

    price_sample = price_sample.sort_values(["date", "ticker"])
    fund_sample = fund_sample.sort_values(["effective_date", "ticker"])

    price_sample.reset_index(drop=True, inplace=True)
    fund_sample.reset_index(drop=True, inplace=True)

    merged = pd.merge_asof(
        price_sample,
        fund_sample,
        left_on='date',  # daily date
        right_on='effective_date',  # fundamentals date
        by='ticker',
        direction='backward'  # use the most recent fund data <= daily date
    )

    merged.to_csv("merged.csv", index=False)

    return merged


"""

Below are helper funcations

"""

def check_ticker_count(df):
    return len(df.ticker.unique())

def filter_columns(df):

    df = df[[
        'ticker',
        'adj_close',
        'date',
        'daily_roi',
        'daily P/E',
        'daily D/M',

    ]]

    return df


def analyze_ticker(stock_analysis, ticker):

    stock_analysis = stock_analysis[stock_analysis['ticker'] == ticker]
    stock_analysis.dropna(inplace=True)

    return stock_analysis

