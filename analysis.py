import pandas as pd
from Quandl import *
import Quandl.Quandl_Python_Tables_API as quandl_api
from fredapi import Fred
import os
import itertools

import config
import numpy as np



def fred():
    """
    Loads Fred Data for 3 month sofr rate. Subtracts 100 bps to get repo rate
    """
    BPS  = 1.00
    fred = Fred(api_key=config.FRED_API_KEY())

    # Define the date range
    start_date = '2017-01-01'
    end_date = '2025-01-30'

    # Fetch the DTB3 series data
    dtb3_data = fred.get_series('DTB3', observation_start=start_date, observation_end=end_date)

    df = pd.DataFrame(dtb3_data, columns=['Annualized Rate'])
    df.dropna(inplace=True)  # Drop missing values
    df['Adjuated_Rate'] = df['Annualized Rate'] - BPS

    # Convert to daily rates
    df['repo_rate'] = np.maximum(df['Adjuated_Rate'] / 365, -0.0002)  # -2% floor

    df = df[['repo_rate']]

    df.reset_index(inplace=True)
    df.rename(columns={'index':'date'}, inplace=True)

    # Display the data
    return df


import pandas as pd
import numpy as np

def calculate_ladder_weights(n_positions: int) -> np.ndarray:
    """
    Calculate ladder weights that sum to 1.
    Example for 5 positions: [0.333, 0.267, 0.200, 0.133, 0.067]
    """
    steps = np.arange(n_positions, 0, -1)
    weights = steps / steps.sum()
    return weights

def calculate_equal_weights(n_positions: int) -> np.ndarray:
    """
    Calculate equal weights that sum to 1.
    Example for 5 positions: [0.2, 0.2, 0.2, 0.2, 0.2]
    """
    return np.ones(n_positions) / n_positions

def execute_strategy(df: pd.DataFrame,
                    ratio_column: str = 'daily P/E',
                    initial_capital: float = 10_000_000,
                    n_positions: int = 5,
                    retention_threshold: float = 0.1,
                    ascending: bool = True,
                    equal_weight: bool = False) -> tuple[pd.DataFrame, float]:
    """
    Execute a long-short trading strategy based on ranking by specified ratio.

    Args:
        df: DataFrame with columns [date, ticker, adj_close, {ratio_column}, repo_rate]
        ratio_column: Name of column to use for ranking positions
        initial_capital: Starting capital for the strategy
        n_positions: Number of positions to take on each side
        retention_threshold: How close to the cutoff a stock needs to be to retain position
        ascending: If True, shorts will be lowest ratio stocks. If False, shorts will be highest ratio stocks
        equal_weight: If True, use equal weighting; if False, use ladder weighting
    """
    capital = initial_capital
    portfolio = {}
    performance_data = []

    # Calculate position weights based on weighting scheme
    position_weights = calculate_equal_weights(n_positions) if equal_weight else calculate_ladder_weights(n_positions)

    # Prepare data
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df = df.sort_values(by=['date', 'ticker'])

    for month, group in df.groupby('month'):
        # Get end-of-month data
        end_of_month_data = group.groupby('ticker').last().reset_index()

        # Handle missing or invalid ratio values
        end_of_month_data[ratio_column] = pd.to_numeric(end_of_month_data[ratio_column], errors='coerce')
        valid_mask = (
            end_of_month_data[ratio_column].notna() &
            (end_of_month_data[ratio_column] != np.inf) &
            (end_of_month_data[ratio_column] > 0)
        )
        end_of_month_data = end_of_month_data[valid_mask].sort_values(ratio_column, ascending=ascending)

        # If no valid stocks this month, skip to next month
        if len(end_of_month_data) == 0:
            continue

        # Calculate rank percentiles for retention threshold
        end_of_month_data['rank_percentile'] = np.arange(len(end_of_month_data)) / len(end_of_month_data)

        # Calculate retention thresholds
        base_percentile = n_positions / len(end_of_month_data)
        short_threshold = base_percentile * (1 + retention_threshold)
        long_threshold = 1 - base_percentile * (1 + retention_threshold)

        # Select stocks including retention threshold zone
        short_stocks = end_of_month_data[end_of_month_data['rank_percentile'] <= short_threshold].head(n_positions).copy()
        long_stocks = end_of_month_data[end_of_month_data['rank_percentile'] >= long_threshold].tail(n_positions).copy()

        # Add weights based on scheme
        if equal_weight:
            short_stocks['weight'] = position_weights
            long_stocks['weight'] = position_weights
        else:
            short_stocks['weight'] = position_weights[::-1]  # Reverse for shorts
            long_stocks['weight'] = position_weights  # Highest weight to highest ratio

        # Monthly totals
        month_realized_pnl = 0
        month_repo_costs = 0
        month_notional_traded = 0
        positions_to_close = []

        # Calculate end of month P&L for all positions
        for ticker, pos in portfolio.items():
            stock_data = end_of_month_data[end_of_month_data['ticker'] == ticker]
            stock_month_data = group[group['ticker'] == ticker]

            # Get end of month price and check position retention
            if len(stock_data) > 0:
                current_price = stock_data['adj_close'].iloc[0]
                rank_percentile = stock_data['rank_percentile'].iloc[0]

                # Determine if position should be closed based on retention threshold
                if pos['side'] == 'long':
                    should_close = rank_percentile < long_threshold
                else:  # short position
                    should_close = rank_percentile > short_threshold
            elif len(stock_month_data) > 0:
                current_price = stock_month_data['adj_close'].iloc[-1]
                should_close = True
            else:
                print(f"Warning: Stock {ticker} disappeared - using entry price for closing")
                current_price = pos['entry_price']
                should_close = True

            # Add to monthly P&L
            pnl = (current_price - pos['entry_price']) * pos['shares']
            month_realized_pnl += pnl

            # Add repo costs for shorts
            if pos['side'] == 'short':
                repo_cost = abs(pos['shares']) * pos['entry_price'] * pos['repo_rate']
                month_repo_costs += repo_cost

            if should_close:
                positions_to_close.append(ticker)

        # Update capital with month's total P&L and costs
        capital += month_realized_pnl - month_repo_costs

        # Remove closed positions
        for ticker in positions_to_close:
            del portfolio[ticker]

        # Open/adjust positions with selected weighting
        for stocks, side in [(long_stocks, 'long'), (short_stocks, 'short')]:
            for _, row in stocks.iterrows():
                position_capital = capital * row['weight'] / 2  # Divide by 2 as we split capital between long/short
                shares = int(position_capital // row['adj_close'])

                if shares == 0:
                    continue

                if side == 'short':
                    shares = -shares

                if row['ticker'] in portfolio:
                    # Update existing position size
                    old_pnl = (row['adj_close'] - portfolio[row['ticker']]['entry_price']) * portfolio[row['ticker']]['shares']
                    month_realized_pnl += old_pnl
                    if side == 'short':
                        old_repo = abs(portfolio[row['ticker']]['shares']) * portfolio[row['ticker']]['entry_price'] * portfolio[row['ticker']]['repo_rate']
                        month_repo_costs += old_repo

                month_notional_traded += abs(shares) * row['adj_close']


                portfolio[row['ticker']] = {
                    'shares': shares,
                    'entry_price': row['adj_close'],
                    'repo_rate': row['repo_rate'],
                    'side': side,
                    'weight': row['weight']
                }

        # # Store monthly summary stats
        # print(f"\nMonth: {month}")
        # print(f"Sum Realized PnL: {month_realized_pnl:,.2f}")
        # print(f"Sum Costs: {month_repo_costs:,.2f}")
        # print(f"Capital: {capital:,.2f}")
        # print(f"Num Positions: {len(portfolio)}")
        # print(f"  Long: {sum(1 for pos in portfolio.values() if pos['side'] == 'long')}")
        # print(f"  Short: {sum(1 for pos in portfolio.values() if pos['side'] == 'short')}"),

        performance_data.append({
            'month': month,
            'realized_pnl': month_realized_pnl,
            'costs': month_repo_costs,
            'capital': capital,
            'return': (capital - initial_capital) / initial_capital,
            'num_positions': len(portfolio),
            'num_long': sum(1 for pos in portfolio.values() if pos['side'] == 'long'),
            'num_short': sum(1 for pos in portfolio.values() if pos['side'] == 'short'),
            'notional_traded': month_notional_traded

        })



    performance_data = pd.DataFrame(performance_data)
    performance_data['monthly_returns'] = performance_data['capital'].pct_change()
    performance_data = performance_data[performance_data['month'] < '2024-07-31']

    return pd.DataFrame(performance_data), capital


import pandas as pd
import numpy as np


def execute_strategy_flex(df: pd.DataFrame,
                    ratio_column: str = 'daily P/E',
                    initial_capital: float = 10_000_000,
                    long_short_ratio: float = 0.1,  # Back to using ratio instead of fixed N
                    retention_threshold: float = 0.1,
                    ascending: bool = True,
                    equal_weight: bool = False) -> tuple[pd.DataFrame, float]:
    """
    Execute a long-short trading strategy based on ranking by specified ratio.

    Args:
        df: DataFrame with columns [date, ticker, adj_close, {ratio_column}, repo_rate]
        ratio_column: Name of column to use for ranking positions
        initial_capital: Starting capital for the strategy
        long_short_ratio: Percentage of universe for long/short positions (e.g., 0.1 for top/bottom 10%)
        retention_threshold: How close to the cutoff a stock needs to be to retain position
        ascending: If True, shorts will be lowest ratio stocks. If False, shorts will be highest ratio stocks
        equal_weight: If True, use equal weighting; if False, use ladder weighting
    """
    capital = initial_capital
    portfolio = {}
    performance_data = []

    # Prepare data
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df = df.sort_values(by=['date', 'ticker'])

    for month, group in df.groupby('month'):
        # Get end-of-month data
        end_of_month_data = group.groupby('ticker').last().reset_index()

        # Handle missing or invalid ratio values
        end_of_month_data[ratio_column] = pd.to_numeric(end_of_month_data[ratio_column], errors='coerce')
        valid_mask = (
            end_of_month_data[ratio_column].notna() &
            (end_of_month_data[ratio_column] != np.inf) &
            (end_of_month_data[ratio_column] > 0)
        )
        end_of_month_data = end_of_month_data[valid_mask].sort_values(ratio_column, ascending=ascending)

        # If no valid stocks this month, skip to next month
        if len(end_of_month_data) == 0:
            continue

        # Calculate rank percentiles for retention threshold
        end_of_month_data['rank_percentile'] = np.arange(len(end_of_month_data)) / len(end_of_month_data)

        # Calculate number of positions and thresholds
        n_positions = int(len(end_of_month_data) * long_short_ratio)
        n_positions = max(1, min(n_positions, len(end_of_month_data) // 2))  # At least 1, at most half the universe

        # Calculate retention thresholds
        short_threshold = long_short_ratio * (1 + retention_threshold)
        long_threshold = 1 - long_short_ratio * (1 + retention_threshold)

        # Select stocks including retention threshold zone
        short_stocks = end_of_month_data[end_of_month_data['rank_percentile'] <= short_threshold].head(n_positions).copy()
        long_stocks = end_of_month_data[end_of_month_data['rank_percentile'] >= long_threshold].tail(n_positions).copy()

        # Calculate weights for this month's positions
        position_weights = calculate_equal_weights(n_positions) if equal_weight else calculate_ladder_weights(n_positions)

        # Add weights based on scheme
        if equal_weight:
            short_stocks['weight'] = position_weights
            long_stocks['weight'] = position_weights
        else:
            short_stocks['weight'] = position_weights[::-1]  # Reverse for shorts
            long_stocks['weight'] = position_weights  # Highest weight to highest ratio

        # Monthly totals
        month_realized_pnl = 0
        month_repo_costs = 0
        month_notional_traded = 0
        positions_to_close = []

        # Calculate end of month P&L for all positions
        for ticker, pos in portfolio.items():
            stock_data = end_of_month_data[end_of_month_data['ticker'] == ticker]
            stock_month_data = group[group['ticker'] == ticker]

            # Get end of month price and check position retention
            if len(stock_data) > 0:
                current_price = stock_data['adj_close'].iloc[0]
                rank_percentile = stock_data['rank_percentile'].iloc[0]

                # Determine if position should be closed based on retention threshold
                if pos['side'] == 'long':
                    should_close = rank_percentile < long_threshold
                else:  # short position
                    should_close = rank_percentile > short_threshold
            elif len(stock_month_data) > 0:
                current_price = stock_month_data['adj_close'].iloc[-1]
                should_close = True
            else:
                print(f"Warning: Stock {ticker} disappeared - using entry price for closing")
                current_price = pos['entry_price']
                should_close = True

            # Add to monthly P&L
            pnl = (current_price - pos['entry_price']) * pos['shares']
            month_realized_pnl += pnl

            # Add repo costs for shorts
            if pos['side'] == 'short':
                repo_cost = abs(pos['shares']) * pos['entry_price'] * pos['repo_rate']
                month_repo_costs += repo_cost

            if should_close:
                positions_to_close.append(ticker)

        # Update capital with month's total P&L and costs
        capital += month_realized_pnl - month_repo_costs

        # Remove closed positions
        for ticker in positions_to_close:
            del portfolio[ticker]

        # Open/adjust positions with selected weighting
        for stocks, side in [(long_stocks, 'long'), (short_stocks, 'short')]:
            for _, row in stocks.iterrows():
                position_capital = capital * row['weight'] / 2  # Divide by 2 as we split capital between long/short
                shares = int(position_capital // row['adj_close'])

                if shares == 0:
                    continue

                if side == 'short':
                    shares = -shares

                if row['ticker'] in portfolio:
                    # Update existing position size
                    old_pnl = (row['adj_close'] - portfolio[row['ticker']]['entry_price']) * portfolio[row['ticker']]['shares']
                    month_realized_pnl += old_pnl
                    if side == 'short':
                        old_repo = abs(portfolio[row['ticker']]['shares']) * portfolio[row['ticker']]['entry_price'] * portfolio[row['ticker']]['repo_rate']
                        month_repo_costs += old_repo


                month_notional_traded += abs(shares) * row['adj_close']


                portfolio[row['ticker']] = {
                    'shares': shares,
                    'entry_price': row['adj_close'],
                    'repo_rate': row['repo_rate'],
                    'side': side,
                    'weight': row['weight']
                }
        #
        # # Store monthly summary stats
        # print(f"\nMonth: {month}")
        # print(f"Sum Realized PnL: {month_realized_pnl:,.2f}")
        # print(f"Sum Costs: {month_repo_costs:,.2f}")
        # print(f"Capital: {capital:,.2f}")
        # print(f"Number of stocks in universe: {len(end_of_month_data)}")
        # print(f"Target positions per side: {n_positions}")
        # print(f"Actual positions: {len(portfolio)}")
        # print(f"  Long: {sum(1 for pos in portfolio.values() if pos['side'] == 'long')}")
        # print(f"  Short: {sum(1 for pos in portfolio.values() if pos['side'] == 'short')}")

        performance_data.append({
            'month': month,
            'realized_pnl': month_realized_pnl,
            'costs': month_repo_costs,
            'capital': capital,
            'return': (capital - initial_capital) / initial_capital,
            'universe_size': len(end_of_month_data),
            'target_positions': n_positions,
            'num_positions': len(portfolio),
            'num_long': sum(1 for pos in portfolio.values() if pos['side'] == 'long'),
            'num_short': sum(1 for pos in portfolio.values() if pos['side'] == 'short'),
            'notional_traded': month_notional_traded  # New column added

        })

    performance_data = pd.DataFrame(performance_data)
    performance_data['monthly_returns'] = performance_data['capital'].pct_change()

    return pd.DataFrame(performance_data), capital


def execute_strategy_multiple_ratio(df: pd.DataFrame,
                     ratio_columns: list[str] = ['daily P/E', 'P/B'],  # Changed to accept list of ratios
                     initial_capital: float = 10_000_000,
                     n_positions: int = 5,
                     retention_threshold: float = 0.1,
                     ascending: bool = True,
                     equal_weight: bool = False) -> tuple[pd.DataFrame, float]:
    """
    Execute a long-short trading strategy based on ranking by multiple specified ratios.

    Args:
        df: DataFrame with columns [date, ticker, adj_close, {ratio_columns}, repo_rate]
        ratio_columns: List of column names to use for ranking positions
        initial_capital: Starting capital for the strategy
        n_positions: Number of positions to take on each side
        retention_threshold: How close to the cutoff a stock needs to be to retain position
        ascending: If True, shorts will be lowest combined ratio stocks. If False, shorts will be highest combined ratio stocks
        equal_weight: If True, use equal weighting; if False, use ladder weighting
    """
    capital = initial_capital
    portfolio = {}
    performance_data = []

    # Calculate position weights based on weighting scheme
    position_weights = calculate_equal_weights(n_positions) if equal_weight else calculate_ladder_weights(n_positions)

    # Prepare data
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df = df.sort_values(by=['date', 'ticker'])

    for month, group in df.groupby('month'):
        # Get end-of-month data
        end_of_month_data = group.groupby('ticker').last().reset_index()

        # Handle missing or invalid ratio values for all ratio columns
        valid_mask = pd.Series(True, index=end_of_month_data.index)

        for ratio_column in ratio_columns:
            end_of_month_data[ratio_column] = pd.to_numeric(end_of_month_data[ratio_column], errors='coerce')
            column_mask = (
                    end_of_month_data[ratio_column].notna() &
                    (end_of_month_data[ratio_column] != np.inf) &
                    (end_of_month_data[ratio_column] > 0)
            )
            valid_mask &= column_mask

        end_of_month_data = end_of_month_data[valid_mask]

        # If no valid stocks this month, skip to next month
        if len(end_of_month_data) == 0:
            continue

        # Calculate normalized ranks for each ratio
        for ratio_column in ratio_columns:
            rank_col = f'{ratio_column}_rank'
            end_of_month_data[rank_col] = end_of_month_data[ratio_column].rank(ascending=ascending)
            # Normalize ranks to 0-1 scale
            end_of_month_data[rank_col] = (end_of_month_data[rank_col] - 1) / (len(end_of_month_data) - 1)

        # Calculate combined rank (average of normalized ranks)
        rank_columns = [f'{ratio_column}_rank' for ratio_column in ratio_columns]
        end_of_month_data['combined_rank'] = end_of_month_data[rank_columns].mean(axis=1)

        # Sort by combined rank and calculate percentiles
        end_of_month_data = end_of_month_data.sort_values('combined_rank', ascending=ascending)
        end_of_month_data['rank_percentile'] = np.arange(len(end_of_month_data)) / len(end_of_month_data)

        # Calculate retention thresholds
        base_percentile = n_positions / len(end_of_month_data)
        short_threshold = base_percentile * (1 + retention_threshold)
        long_threshold = 1 - base_percentile * (1 + retention_threshold)

        # Select stocks including retention threshold zone
        short_stocks = end_of_month_data[end_of_month_data['rank_percentile'] <= short_threshold].head(
            n_positions).copy()
        long_stocks = end_of_month_data[end_of_month_data['rank_percentile'] >= long_threshold].tail(n_positions).copy()

        # Add weights based on scheme
        if equal_weight:
            short_stocks['weight'] = position_weights
            long_stocks['weight'] = position_weights
        else:
            short_stocks['weight'] = position_weights[::-1]  # Reverse for shorts
            long_stocks['weight'] = position_weights  # Highest weight to highest ratio

        # Monthly totals
        month_realized_pnl = 0
        month_repo_costs = 0
        positions_to_close = []
        month_notional_traded = 0

        # Calculate end of month P&L for all positions
        for ticker, pos in portfolio.items():
            stock_data = end_of_month_data[end_of_month_data['ticker'] == ticker]
            stock_month_data = group[group['ticker'] == ticker]

            # Get end of month price and check position retention
            if len(stock_data) > 0:
                current_price = stock_data['adj_close'].iloc[0]
                rank_percentile = stock_data['rank_percentile'].iloc[0]

                # Determine if position should be closed based on retention threshold
                if pos['side'] == 'long':
                    should_close = rank_percentile < long_threshold
                else:  # short position
                    should_close = rank_percentile > short_threshold
            elif len(stock_month_data) > 0:
                current_price = stock_month_data['adj_close'].iloc[-1]
                should_close = True
            else:
                print(f"Warning: Stock {ticker} disappeared - using entry price for closing")
                current_price = pos['entry_price']
                should_close = True

            # Add to monthly P&L
            pnl = (current_price - pos['entry_price']) * pos['shares']
            month_realized_pnl += pnl

            # Add repo costs for shorts
            if pos['side'] == 'short':
                repo_cost = abs(pos['shares']) * pos['entry_price'] * pos['repo_rate']
                month_repo_costs += repo_cost

            if should_close:
                positions_to_close.append(ticker)

        # Update capital with month's total P&L and costs
        capital += month_realized_pnl - month_repo_costs

        # Remove closed positions
        for ticker in positions_to_close:
            del portfolio[ticker]

        # Open/adjust positions with selected weighting
        for stocks, side in [(long_stocks, 'long'), (short_stocks, 'short')]:
            for _, row in stocks.iterrows():
                position_capital = capital * row['weight'] / 2  # Divide by 2 as we split capital between long/short
                shares = int(position_capital // row['adj_close'])

                if shares == 0:
                    continue

                if side == 'short':
                    shares = -shares

                if row['ticker'] in portfolio:
                    # Update existing position size
                    old_pnl = (row['adj_close'] - portfolio[row['ticker']]['entry_price']) * portfolio[row['ticker']][
                        'shares']
                    month_realized_pnl += old_pnl
                    if side == 'short':
                        old_repo = abs(portfolio[row['ticker']]['shares']) * portfolio[row['ticker']]['entry_price'] * \
                                   portfolio[row['ticker']]['repo_rate']
                        month_repo_costs += old_repo

                month_notional_traded += abs(shares) * row['adj_close']

                portfolio[row['ticker']] = {
                    'shares': shares,
                    'entry_price': row['adj_close'],
                    'repo_rate': row['repo_rate'],
                    'side': side,
                    'weight': row['weight']
                }

        # Store monthly summary stats
        # print(f"\nMonth: {month}")
        # print(f"Sum Realized PnL: {month_realized_pnl:,.2f}")
        # print(f"Sum Costs: {month_repo_costs:,.2f}")
        # print(f"Capital: {capital:,.2f}")
        # print(f"Num Positions: {len(portfolio)}")
        # print(f"  Long: {sum(1 for pos in portfolio.values() if pos['side'] == 'long')}")
        # print(f"  Short: {sum(1 for pos in portfolio.values() if pos['side'] == 'short')}")

        performance_data.append({
            'month': month,
            'realized_pnl': month_realized_pnl,
            'costs': month_repo_costs,
            'capital': capital,
            'return': (capital - initial_capital) / initial_capital,
            'num_positions': len(portfolio),
            'num_long': sum(1 for pos in portfolio.values() if pos['side'] == 'long'),
            'num_short': sum(1 for pos in portfolio.values() if pos['side'] == 'short'),
            'notional_traded': month_notional_traded  # New column added
        })

    performance_data = pd.DataFrame(performance_data)
    performance_data['monthly_returns'] = performance_data['capital'].pct_change()

    return performance_data, capital


def execute_strategy_ratio_change(df: pd.DataFrame,
                     ratio_column: str = 'daily P/E',
                     initial_capital: float = 10_000_000,
                     n_positions: int = 5,
                     retention_threshold: float = 0.1,
                     ascending: bool = True,
                     equal_weight: bool = False,
                     lookback_periods: int = 1) -> tuple[pd.DataFrame, float]:
    """
    Execute a long-short trading strategy based on changes in fundamental ratios.

    Args:
        df: DataFrame with columns [date, ticker, adj_close, {ratio_column}, repo_rate]
        ratio_column: Name of column to use for ranking positions
        initial_capital: Starting capital for the strategy
        n_positions: Number of positions to take on each side
        retention_threshold: How close to the cutoff a stock needs to be to retain position
        ascending: If True, shorts will be stocks with largest ratio decrease
        lookback_periods: Number of months to look back for calculating changes
    """
    capital = initial_capital
    portfolio = {}
    performance_data = []

    # Calculate position weights based on weighting scheme
    position_weights = calculate_equal_weights(n_positions) if equal_weight else calculate_ladder_weights(n_positions)

    # Prepare data
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df = df.sort_values(by=['date', 'ticker'])

    for month, group in df.groupby('month'):
        # Get end-of-month data
        end_of_month_data = group.groupby('ticker').last().reset_index()

        # Get previous period's data for ratio change calculation
        previous_month = month - lookback_periods
        previous_data = df[df['month'] == previous_month].groupby('ticker').last().reset_index()

        # Calculate ratio changes
        end_of_month_data = end_of_month_data.merge(
            previous_data[['ticker', ratio_column]],
            on='ticker',
            how='left',
            suffixes=('', '_prev')
        )

        # Calculate percentage change in ratio
        change_col = f'{ratio_column}_change'
        end_of_month_data[change_col] = (
                (end_of_month_data[ratio_column] - end_of_month_data[f'{ratio_column}_prev']) /
                end_of_month_data[f'{ratio_column}_prev']
        )

        # Handle missing or invalid ratio values
        valid_mask = (
                end_of_month_data[change_col].notna() &
                (end_of_month_data[change_col] != np.inf) &
                (end_of_month_data[change_col] != -np.inf)
        )
        end_of_month_data = end_of_month_data[valid_mask].sort_values(change_col, ascending=ascending)

        # If no valid stocks this month, skip to next month
        if len(end_of_month_data) == 0:
            continue

        # Calculate rank percentiles for retention threshold
        end_of_month_data['rank_percentile'] = np.arange(len(end_of_month_data)) / len(end_of_month_data)

        # Calculate retention thresholds
        base_percentile = n_positions / len(end_of_month_data)
        short_threshold = base_percentile * (1 + retention_threshold)
        long_threshold = 1 - base_percentile * (1 + retention_threshold)

        # Select stocks including retention threshold zone
        short_stocks = end_of_month_data[end_of_month_data['rank_percentile'] <= short_threshold].head(
            n_positions).copy()
        long_stocks = end_of_month_data[end_of_month_data['rank_percentile'] >= long_threshold].tail(n_positions).copy()

        # Add weights based on scheme
        if equal_weight:
            short_stocks['weight'] = position_weights
            long_stocks['weight'] = position_weights
        else:
            short_stocks['weight'] = position_weights[::-1]  # Reverse for shorts
            long_stocks['weight'] = position_weights  # Highest weight to highest ratio change

        # Monthly totals
        month_realized_pnl = 0
        month_repo_costs = 0
        positions_to_close = []
        month_notional_traded = 0

        # Calculate end of month P&L for all positions
        for ticker, pos in portfolio.items():
            stock_data = end_of_month_data[end_of_month_data['ticker'] == ticker]
            stock_month_data = group[group['ticker'] == ticker]

            # Get end of month price and check position retention
            if len(stock_data) > 0:
                current_price = stock_data['adj_close'].iloc[0]
                rank_percentile = stock_data['rank_percentile'].iloc[0]

                # Determine if position should be closed based on retention threshold
                if pos['side'] == 'long':
                    should_close = rank_percentile < long_threshold
                else:  # short position
                    should_close = rank_percentile > short_threshold
            elif len(stock_month_data) > 0:
                current_price = stock_month_data['adj_close'].iloc[-1]
                should_close = True
            else:
                print(f"Warning: Stock {ticker} disappeared - using entry price for closing")
                current_price = pos['entry_price']
                should_close = True

            # Add to monthly P&L
            pnl = (current_price - pos['entry_price']) * pos['shares']
            month_realized_pnl += pnl

            # Add repo costs for shorts
            if pos['side'] == 'short':
                repo_cost = abs(pos['shares']) * pos['entry_price'] * pos['repo_rate']
                month_repo_costs += repo_cost

            if should_close:
                positions_to_close.append(ticker)

        # Update capital with month's total P&L and costs
        capital += month_realized_pnl - month_repo_costs

        # Remove closed positions
        for ticker in positions_to_close:
            del portfolio[ticker]

        # Open/adjust positions with selected weighting
        for stocks, side in [(long_stocks, 'long'), (short_stocks, 'short')]:
            for _, row in stocks.iterrows():
                position_capital = capital * row['weight'] / 2  # Divide by 2 as we split capital between long/short
                shares = int(position_capital // row['adj_close'])

                if shares == 0:
                    continue

                if side == 'short':
                    shares = -shares

                if row['ticker'] in portfolio:
                    # Update existing position size
                    old_pnl = (row['adj_close'] - portfolio[row['ticker']]['entry_price']) * portfolio[row['ticker']][
                        'shares']
                    month_realized_pnl += old_pnl
                    if side == 'short':
                        old_repo = abs(portfolio[row['ticker']]['shares']) * portfolio[row['ticker']]['entry_price'] * \
                                   portfolio[row['ticker']]['repo_rate']
                        month_repo_costs += old_repo

                month_notional_traded += abs(shares) * row['adj_close']

                portfolio[row['ticker']] = {
                    'shares': shares,
                    'entry_price': row['adj_close'],
                    'repo_rate': row['repo_rate'],
                    'side': side,
                    'weight': row['weight']
                }

        # Store monthly summary stats
        # print(f"\nMonth: {month}")
        # print(f"Sum Realized PnL: {month_realized_pnl:,.2f}")
        # print(f"Sum Costs: {month_repo_costs:,.2f}")
        # print(f"Capital: {capital:,.2f}")
        # print(f"Num Positions: {len(portfolio)}")
        # print(f"  Long: {sum(1 for pos in portfolio.values() if pos['side'] == 'long')}")
        # print(f"  Short: {sum(1 for pos in portfolio.values() if pos['side'] == 'short')}")

        performance_data.append({
            'month': month,
            'realized_pnl': month_realized_pnl,
            'costs': month_repo_costs,
            'capital': capital,
            'return': (capital - initial_capital) / initial_capital,
            'num_positions': len(portfolio),
            'num_long': sum(1 for pos in portfolio.values() if pos['side'] == 'long'),
            'num_short': sum(1 for pos in portfolio.values() if pos['side'] == 'short'),
            'notional_traded': month_notional_traded  # New column added
        })

    performance_data = pd.DataFrame(performance_data)
    performance_data['monthly_returns'] = performance_data['capital'].pct_change()

    return performance_data, capital


# def optimize_strategy(
#         strategy_func,  # The strategy function to optimize
#         df: pd.DataFrame,  # Input data
#         param_ranges: dict,  # Dictionary of parameters and their ranges
#         fixed_params: dict = None,  # Fixed parameters that don't change
#         n_steps: dict = None,  # Number of steps for each parameter
#         objective='sharpe'  # Metric to optimize ('sharpe', 'return', 'drawdown')
# ) -> dict:
#     """
#     Generic strategy optimizer using grid search.
#
#     Args:
#         strategy_func: Function that implements the trading strategy
#         df: Input DataFrame
#         param_ranges: Dict of parameter ranges, e.g. {'n_positions': (3, 10)}
#         fixed_params: Dict of fixed parameters
#         n_steps: Dict of number of steps for each parameter
#         objective: Metric to optimize for
#
#     Returns:
#         Dictionary containing optimal parameters and all results
#     """
#     fixed_params = fixed_params or {}
#     n_steps = n_steps or {}
#
#     # Generate parameter combinations
#     param_values = {}
#     for param, (start, end) in param_ranges.items():
#         if isinstance(start, int) and isinstance(end, int):
#             # Integer parameters
#             param_values[param] = range(start, end + 1)
#         else:
#             # Float parameters
#             steps = n_steps.get(param, 20)
#             param_values[param] = np.linspace(start, end, steps)
#
#     # Calculate total combinations
#     total_combinations = np.prod([len(values) for values in param_values.values()])
#     current = 0
#
#     # Store results
#     best_metric = -np.inf if objective != 'drawdown' else np.inf
#     best_params = None
#     results = []
#
#     # Generate all combinations of parameters
#     param_names = list(param_values.keys())
#     for params in itertools.product(*param_values.values()):
#         current += 1
#         param_dict = dict(zip(param_names, params))
#         param_dict.update(fixed_params)  # Add fixed parameters
#
#         print(f"\nTesting combination {current}/{total_combinations}")
#         print("Parameters:", param_dict)
#
#         try:
#             # Run strategy with current parameters
#             perf_data, final_capital = strategy_func(df=df, **param_dict)
#
#             # Calculate performance metrics
#             returns = perf_data['monthly_returns'].dropna()
#             if len(returns) > 0:
#                 metrics = calculate_metrics(returns, perf_data['capital'],
#                                             final_capital, param_dict['initial_capital'])
#
#                 # Store all results
#                 result_dict = {**param_dict, **metrics}
#                 results.append(result_dict)
#
#                 # Update best parameters based on objective
#                 current_metric = metrics[objective]
#                 is_better = (
#                     current_metric > best_metric if objective != 'drawdown'
#                     else current_metric < best_metric
#                 )
#
#                 if is_better:
#                     best_metric = current_metric
#                     best_params = result_dict
#
#                 print(f"Metrics: {metrics}")
#
#         except Exception as e:
#             print(f"Error with parameters {param_dict}: {str(e)}")
#             continue
#
#     return {
#         'best_params': best_params,
#         'all_results': pd.DataFrame(results)
#     }


def calculate_metrics(returns: pd.Series,
                      capital: pd.Series,
                      final_capital: float,
                      initial_capital: float) -> dict:
    """Calculate common strategy performance metrics."""
    sharpe = np.sqrt(12) * returns.mean() / returns.std()
    total_return = (final_capital - initial_capital) / initial_capital

    # Calculate drawdown
    roll_max = capital.expanding().max()
    drawdowns = capital / roll_max - 1
    max_drawdown = drawdowns.min()

    # Calculate other metrics
    volatility = returns.std() * np.sqrt(12)
    sortino = np.sqrt(12) * returns.mean() / returns[returns < 0].std()

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'sortino': sortino,
        'final_capital': final_capital
    }




import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

def plot_strategy_performance(performance_data: pd.DataFrame):
    """
    Plots the performance of the trading strategy over time using Seaborn.

    Args:
        performance_data (pd.DataFrame): DataFrame containing at least ['month', 'capital'] columns.
    """
    # Convert Period to Timestamp if necessary
    if isinstance(performance_data['month'].dtype, pd.PeriodDtype):
        performance_data['month'] = performance_data['month'].dt.to_timestamp()

    # Ensure 'month' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(performance_data['month']):
        performance_data['month'] = pd.to_datetime(performance_data['month'])

    # Drop any NaN values in 'month' or 'capital'
    performance_data = performance_data.dropna(subset=['month', 'capital'])

    # Set Seaborn style
    sns.set(style="darkgrid")

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=performance_data, x='month', y='capital', marker='o', label='Capital Over Time')

    # Formatting
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.title("Strategy Performance Over Time")
    plt.legend()
    plt.xticks(rotation=45)

    # Format x-axis for better readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Adjust interval if needed

    # Show plot
    plt.show()


def plot_multiple_strategies(strategy_data_list: list, strategy_names: list):
    """
    Plots multiple strategy performances side by side.

    Args:
        strategy_data_list (list of pd.DataFrame): List of DataFrames, each containing ['month', 'capital'].
        strategy_names (list of str): List of names corresponding to each strategy.
    """
    num_strategies = len(strategy_data_list)
    fig, axes = plt.subplots(1, num_strategies, figsize=(6 * num_strategies, 6), sharey=True)

    # Ensure axes is iterable even for a single subplot
    if num_strategies == 1:
        axes = [axes]

    for i, (performance_data, name) in enumerate(zip(strategy_data_list, strategy_names)):
        # Ensure 'month' is in datetime format
        if isinstance(performance_data['month'].dtype, pd.PeriodDtype):
            performance_data['month'] = performance_data['month'].dt.to_timestamp()
        if not pd.api.types.is_datetime64_any_dtype(performance_data['month']):
            performance_data['month'] = pd.to_datetime(performance_data['month'])

        sns.lineplot(ax=axes[i], data=performance_data, x='month', y='capital', label=name)
        axes[i].set_title(f"{name} Performance")
        axes[i].set_xlabel("Date")
        if i == 0:  # Only show y-axis label on the first plot
            axes[i].set_ylabel("Capital ($)")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def plot_overlayed_strategies(strategy_data_list: list, strategy_names: list):
    """
    Overlays multiple strategy performances on a single plot.

    Args:
        strategy_data_list (list of pd.DataFrame): List of DataFrames, each containing ['month', 'capital'].
        strategy_names (list of str): List of names corresponding to each strategy.
    """
    # Set Seaborn style
    sns.set(style="darkgrid")

    plt.figure(figsize=(12, 6))

    for performance_data, name in zip(strategy_data_list, strategy_names):
        # Ensure 'month' is in datetime format
        if isinstance(performance_data['month'].dtype, pd.PeriodDtype):
            performance_data['month'] = performance_data['month'].dt.to_timestamp()
        if not pd.api.types.is_datetime64_any_dtype(performance_data['month']):
            performance_data['month'] = pd.to_datetime(performance_data['month'])

        sns.lineplot(data=performance_data, x='month', y='capital', label=name)

    # Formatting
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.title("Overlayed Strategy Performance")
    plt.legend(title="Strategies")
    plt.xticks(rotation=45)

    # Format x-axis for better readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Adjust interval dynamically

    # Show plot
    plt.tight_layout()
    plt.show()


def calculate_risk_metrics(performance_data: pd.DataFrame) -> dict:
    """
    Computes Sharpe ratios, downside beta, tail metrics, maximum drawdown,
    and comparisons of P&L to traded notional.

    Args:
        performance_data (pd.DataFrame): DataFrame containing ['date', 'capital', 'returns', 'notional_traded'].

    Returns:
        dict: Dictionary containing calculated risk metrics.
    """
    performance_data = performance_data.copy()
    performance_data['returns'] = performance_data['capital'].pct_change().dropna()

    # Sharpe Ratio Calculation
    sharpe_ratio = performance_data['returns'].mean() / performance_data['returns'].std() * np.sqrt(252)

    # Downside Beta Calculation
    market_downside = performance_data[performance_data['returns'] < 0]['returns']
    downside_beta = market_downside.cov(performance_data['returns']) / market_downside.var() if len(
        market_downside) > 0 else np.nan

    # Maximum Drawdown Calculation
    cumulative_capital = performance_data['capital'].cummax()
    drawdown = (performance_data['capital'] - cumulative_capital) / cumulative_capital
    max_drawdown = drawdown.min()

    # Tail Risk Metrics (5% VaR & CVaR)
    var_5 = np.percentile(performance_data['returns'].dropna(), 5)
    cvar_5 = performance_data['returns'][performance_data['returns'] <= var_5].mean()

    # P&L to Notional Comparison
    pl_to_notional = (performance_data['capital'].diff() / performance_data['notional_traded']).mean()

    metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'Downside Beta': downside_beta,
        'Max Drawdown': max_drawdown,
        'VaR 5%': var_5,
        'CVaR 5%': cvar_5,
        'P&L to Notional': pl_to_notional
    }

    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    metrics_df.set_index('Metric', inplace=True)

    return metrics_df


import pandas as pd
import numpy as np


def rename_columns(dfs: dict, mapping: dict) -> dict:
    """
    Renames the columns of each DataFrame in `dfs` by prepending the corresponding name from `mapping`.

    Args:
        dfs (dict): Dictionary mapping string names to DataFrames.
        mapping (dict): Dictionary mapping string names to descriptive labels.

    Returns:
        dict: Dictionary of DataFrames with renamed columns.
    """
    renamed_dfs = {}
    for df_name, df in dfs.items():
        renamed_dfs[df_name] = df.rename(columns={col: f"{mapping[df_name]} - {col}" for col in df.columns})
    return renamed_dfs


def find_column_by_keyword(df: pd.DataFrame, keyword: str) -> str:
    """
    Finds the renamed column containing the keyword (e.g., "capital", "notional_traded").

    Args:
        df (pd.DataFrame): DataFrame with renamed columns.
        keyword (str): The original column name keyword to search for.

    Returns:
        str: The actual column name in the DataFrame after renaming.
    """
    for col in df.columns:
        if keyword in col.lower():
            return col
    raise KeyError(f"Column containing '{keyword}' not found in DataFrame: {df.columns}")


def calculate_risk_metrics_final(performance_data: pd.DataFrame) -> dict:
    """
    Computes Sharpe ratios, downside beta, tail metrics, maximum drawdown,
    and comparisons of P&L to traded notional.

    Args:
        performance_data (pd.DataFrame): DataFrame with renamed columns.

    Returns:
        dict: Dictionary containing calculated risk metrics.
    """
    # Dynamically find the column names
    capital_col = find_column_by_keyword(performance_data, "capital")
    notional_col = find_column_by_keyword(performance_data, "notional")

    performance_data = performance_data.copy()

    # Compute returns
    performance_data['returns'] = performance_data[capital_col].pct_change().dropna()

    # Sharpe Ratio Calculation
    sharpe_ratio = performance_data['returns'].mean() / performance_data['returns'].std() * np.sqrt(252)

    # Downside Beta Calculation
    market_downside = performance_data[performance_data['returns'] < 0]['returns']
    downside_beta = market_downside.cov(performance_data['returns']) / market_downside.var() if len(
        market_downside) > 0 else np.nan

    # Maximum Drawdown Calculation
    cumulative_capital = performance_data[capital_col].cummax()
    drawdown = (performance_data[capital_col] - cumulative_capital) / cumulative_capital
    max_drawdown = drawdown.min()

    # Tail Risk Metrics (5% VaR & CVaR)
    var_5 = np.percentile(performance_data['returns'].dropna(), 5)
    cvar_5 = performance_data['returns'][performance_data['returns'] <= var_5].mean()

    # P&L to Notional Comparison
    pl_to_notional = (performance_data[capital_col].diff() / performance_data[notional_col]).mean()

    metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'Downside Beta': downside_beta,
        'Max Drawdown': max_drawdown,
        'VaR 5%': var_5,
        'CVaR 5%': cvar_5,
        'P&L to Notional': pl_to_notional
    }

    return metrics


def calculate_risk_metrics_for_all(dfs: dict, mapping: dict) -> pd.DataFrame:
    """
    Calculates risk metrics for all DataFrames in `dfs` and returns a consolidated DataFrame.

    Args:
        dfs (dict): Dictionary mapping string names to DataFrames.
        mapping (dict): Dictionary mapping string names to descriptive labels.

    Returns:
        pd.DataFrame: DataFrame where each column corresponds to a DataFrame’s calculated risk metrics.
    """
    # Rename columns before processing
    renamed_dfs = rename_columns(dfs, mapping)

    # Compute risk metrics for each DataFrame
    metrics_dict = {mapping[df_name]: calculate_risk_metrics_final(df) for df_name, df in renamed_dfs.items()}

    # Convert dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df


def concatenate_returns(dfs: dict, mapping: dict) -> pd.DataFrame:
    """
    Extracts and concatenates the 'returns' column from all DataFrames into a single DataFrame.

    Args:
        dfs (dict): Dictionary mapping string names to DataFrames.
        mapping (dict): Dictionary mapping string names to descriptive labels.

    Returns:
        pd.DataFrame: DataFrame where each column is the returns time series of a strategy.
    """
    renamed_dfs = rename_columns(dfs, mapping)
    returns_df = pd.DataFrame()