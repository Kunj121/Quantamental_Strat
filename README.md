# Quantile-Based Long-Short Trading Strategy

## Overview

This project implements and analyzes weekly and monthly quantile trading strategies using financial ratios. The core approach is a top-and-bottom decile long-short strategy with dynamic position sizing based on ranking changes in fundamental ratios.

## Trading Strategy Framework

### Ratios Analyzed
- Debt to Market Cap
- Return on Investment
- Price to Earnings
- At least one non-trivial combination of these ratios

### Portfolio Parameters
- Initial capital: 10× first month's gross notional
- Zero trading costs assumed
- Fractional shares allowed
- Easy borrowing with repo rate = funding rate - 100 bps
- Portfolio adjusts for realized/unrealized P&L
- Funding rate: rolling 3-month LIBOR/SOFR

### Performance Metrics
- Sharpe ratio and other risk-adjusted metrics
- Downside beta assessment
- Tail risk analysis
- Maximum drawdown calculation
- P&L vs. traded notional comparison

## Data Requirements

### Data Source
- Nasdaq Zacks Fundamentals B dataset
- Tables used: FC, FR, MT, MKTV, SHRC, HDM

### Universe Selection Criteria
- 200+ U.S. equities (2017–2024)
- Market cap ≥ $1M
- Debt/Market Cap > 0.1 at some point
- Excludes automotive, financial, and insurance sectors

## Strategy Enhancements

### Dynamic Positioning
- Positions based on changes in ratios rather than just absolute values
- Position sizing adjustments:
  - Most attractive vigintiles: Double position
  - Untrustworthy outliers: Halve position

## Project Goals

This study aims to evaluate the effectiveness of fundamental factor-based quantitative trading strategies and their impact on risk-adjusted returns. The research focuses on how dynamic position sizing and ratio change analysis can enhance portfolio performance compared to traditional static approaches.
