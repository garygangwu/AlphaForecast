import pandas as pd
import numpy as np
from pathlib import Path
import os
import talib
import argparse

def compute_log_returns(prices):
    """
    Compute log returns for a price series.
    """
    return np.log(prices / prices.shift(1))

def compute_rsi(df, period=14):
    """
    Compute Relative Strength Index (RSI).
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(df, period=20, std_dev=2):
    """
    Compute Bollinger Bands with price-independent features.
    """
    middle_band = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()

    # Price-independent position within Bollinger Bands (0 to 1)
    bb_position = (df['Close'] - middle_band) / (2 * (std * std_dev))

    # Channel width as volatility indicator (normalized by middle line)
    bb_width_pct = (2 * (std * std_dev)) / middle_band

    return {
        'BB_position': bb_position,
        'BB_width_pct': bb_width_pct
    }

def compute_ma_relative_position(df, period=20):
    """
    Compute relative position to Moving Average.
    """
    ma = df['Close'].rolling(window=period).mean()
    return np.log(df['Close'] / ma)

def compute_ma_crossover(df, short_period=20, long_period=50):
    """
    Compute Moving Average crossover signal.
    """
    short_ma = df['Close'].rolling(window=short_period).mean()
    long_ma = df['Close'].rolling(window=long_period).mean()

    # Crossover signal (1 when short MA > long MA, 0 otherwise)
    return (short_ma > long_ma).astype(int)

def compute_vix_features(vix_df):
    """
    Compute VIX-based features.
    """
    features = {}

    # Daily log return of VIX Close
    features['VIX_log_return'] = compute_log_returns(vix_df['Close'])

    # VIX Position vs MA
    vix_ma_20 = vix_df['Close'].rolling(window=20).mean()
    features['VIX_position_vs_MA20'] = np.log(vix_df['Close'] / vix_ma_20)

    # Fear Indicator (Z-score normalization) - 30 days
    vix_mean = vix_df['Close'].rolling(window=30).mean()
    vix_std = vix_df['Close'].rolling(window=30).std()
    features['VIX_fear_zscore_30d'] = (vix_df['Close'] - vix_mean) / vix_std

    return features

def compute_index_spreads(market_data):
    """
    Compute index spreads between major indices.
    """
    spreads = {}

    # Get log returns for each index
    spy_log_return = compute_log_returns(market_data['SPY']['Close'])
    qqq_log_return = compute_log_returns(market_data['QQQ']['Close'])
    dia_log_return = compute_log_returns(market_data['DIA']['Close'])

    # Index spreads
    spreads['QQQ_SPY_spread'] = qqq_log_return - spy_log_return
    spreads['DIA_SPY_spread'] = dia_log_return - spy_log_return
    spreads['QQQ_DIA_spread'] = qqq_log_return - dia_log_return

    return spreads

def process_market_data(input_dir, output_file):
    """
    Process market data and generate technical indicators.
    """
    print("Loading market data...")

    # Load market data
    market_data = {}
    for symbol in ['DIA', 'QQQ', 'SPY', 'VIX']:
        file_path = input_dir / f"{symbol}.csv"
        if file_path.exists():
            market_data[symbol] = pd.read_csv(file_path)
            market_data[symbol]['Date'] = pd.to_datetime(market_data[symbol]['Date'])
            market_data[symbol] = market_data[symbol].sort_values('Date').reset_index(drop=True)
            print(f"  Loaded {symbol}: {len(market_data[symbol])} rows")
        else:
            print(f"  Warning: {symbol}.csv not found")

    if not market_data:
        print("No market data found!")
        return

    # Find common date range
    common_dates = None
    for symbol, df in market_data.items():
        if common_dates is None:
            common_dates = set(df['Date'])
        else:
            common_dates = common_dates.intersection(set(df['Date']))

    common_dates = sorted(list(common_dates))
    print(f"Common date range: {len(common_dates)} days")

    # Filter all data to common dates
    for symbol in market_data:
        market_data[symbol] = market_data[symbol][market_data[symbol]['Date'].isin(common_dates)]
        market_data[symbol] = market_data[symbol].sort_values('Date').reset_index(drop=True)

    # Initialize result DataFrame
    result_df = pd.DataFrame({'Date': common_dates})

    print("\nGenerating technical indicators...")

    # Process each market index (DIA, QQQ, SPY)
    for symbol in ['DIA', 'QQQ', 'SPY']:
        if symbol not in market_data:
            continue

        print(f"  Processing {symbol}...")
        df = market_data[symbol]

        # 1. General Market Returns (Momentum)
        result_df[f'{symbol}_close_log_return'] = compute_log_returns(df['Close'])

        # 2. Market Momentum Indicators (RSI)
        result_df[f'{symbol}_RSI_14d'] = compute_rsi(df, period=14)

        # 3. Market Bollinger Bands (Volatility & Extremes) - only for SPY and QQQ
        if symbol in ['SPY', 'QQQ']:
            bb_dict = compute_bollinger_bands(df, period=20, std_dev=2)
            result_df[f'{symbol}_BB_position'] = bb_dict['BB_position']

        # 4. Moving Average Relative Positioning (Trend strength)
        result_df[f'{symbol}_Close_vs_MA_20d'] = compute_ma_relative_position(df, period=20)

        # 5. Moving Average Crossovers (Trend changes)
        result_df[f'{symbol}_MA_crossover_20_50'] = compute_ma_crossover(df, short_period=20, long_period=50)

    # 6. Relative Strength & Spread
    print("  Computing index spreads...")
    spreads = compute_index_spreads(market_data)
    for spread_name, spread_values in spreads.items():
        result_df[spread_name] = spread_values

    # 7. VIX Features
    if 'VIX' in market_data:
        print("  Computing VIX features...")
        vix_features = compute_vix_features(market_data['VIX'])
        for vix_name, vix_values in vix_features.items():
            result_df[vix_name] = vix_values

    # Save results
    result_df.to_csv(output_file, index=False)

    print(f"\nMarket technical indicators saved to: {output_file}")
    print(f"Total features: {len(result_df.columns) - 1} (excluding Date)")
    print(f"Date range: {result_df['Date'].min()} to {result_df['Date'].max()}")
    print(f"Total rows: {len(result_df)}")

    # Show feature summary
    print("\nFeature categories:")
    print("  General Market Returns (Momentum): 3 features")
    print("  Volatility & Fear Indicators (VIX): 3 features")
    print("  Market Momentum Indicators (RSI): 3 features")
    print("  Relative Strength & Spread: 3 features")
    print("  Market Bollinger Bands (Volatility & Extremes): 2 features")
    print("  Moving Average Relative Positioning (Trend strength): 3 features")
    print("  Moving Average Crossovers (Trend changes): 3 features")
    print(f"  Total: 20 market technical indicators")

def main():
    """
    Main function to generate market technical indicators.
    """
    parser = argparse.ArgumentParser(description='Generate market technical indicators from market data.')
    parser.add_argument('--input_dir', type=str, default='../market_data',
                       help='Input directory containing market data files')
    parser.add_argument('--output_file', type=str, default='../market_data/market_technical_indicators.csv',
                       help='Output file for market technical indicators')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(exist_ok=True)

    print("="*60)
    print("MARKET TECHNICAL INDICATORS GENERATION")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print("="*60)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return

    process_market_data(input_dir, output_file)

if __name__ == "__main__":
    main()
