import pandas as pd
import numpy as np
from pathlib import Path
import os
import talib
import argparse

def compute_log_returns(prices):
    """
    Compute log returns for a price series.

    Args:
        prices (pd.Series): Price series

    Returns:
        pd.Series: Log returns
    """
    return np.log(prices / prices.shift(1))

def compute_forward_returns(prices, days):
    """
    Compute forward log returns for a given number of days.

    Args:
        prices (pd.Series): Price series
        days (int): Number of days to look forward

    Returns:
        pd.Series: Forward log returns
    """
    # Forward log return: log(P_{t+days} / P_t)
    # This is the standard definition for forward log returns.
    return np.log(prices.shift(-days) / prices)

def compute_moving_averages_relative_to_close(df, periods=[5, 20]):
    """
    Compute moving averages for specified periods relative to Close.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        periods (list): List of periods for moving averages

    Returns:
        dict: Dictionary with moving averages
    """
    ma_dict = {}
    for period in periods:
        ma_dict[f'Close_vs_MA5_{period}d'] = np.log(df['Close'] / df['Close'].rolling(window=period).mean())
    return ma_dict

def compute_rsi(df, period=14):
    """
    Compute Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for RSI calculation

    Returns:
        pd.Series: RSI values
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd_from_log_returns(df, fast=12, slow=26, signal=9):
    """
    Compute MACD (Moving Average Convergence Divergence) from log returns.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period

    Returns:
        dict: Dictionary with MACD line, signal line, and histogram
    """
    ema_fast = df['Close_log_return'].ewm(span=fast).mean()
    ema_slow = df['Close_log_return'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        'MACD_line': macd_line,
        'MACD_signal': signal_line,
        'MACD_histogram': histogram
    }

def compute_normalized_bollinger_bands(df, period=20, std_dev=2):
    """
    Compute Bollinger Bands.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for moving average
        std_dev (int): Number of standard deviations

    Returns:
        dict: Dictionary with upper band, middle band, and lower band
    """
    middle_band = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    # upper_band = middle_band + (std * std_dev)
    # lower_band = middle_band - (std * std_dev)
    # upper_band - lower_band == 2 * (std * std_dev)
    bb_position = (df['Close'] - middle_band) / (2 * (std * std_dev))

    return {
        'BB_position': bb_position
    }

def compute_atr(df, period=14):
    """
    Compute Average True Range (ATR).

    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    return atr

def compute_obv_zscore(df, rolling_window=20):
    """
    Compute On-Balance Volume (OBV) z-score.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.Series: OBV values
    """
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['Volume'].iloc[0]

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return (obv - obv.rolling(window=rolling_window).mean()) \
                   / obv.rolling(window=rolling_window).std()

def compute_vwap(df):
    """
    Compute Volume Weighted Average Price (VWAP).

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.Series: VWAP values
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def compute_relative_volume_moving_averages(df, periods=[5, 20]):
    """
    Compute volume moving averages for specified periods.

    Args:
        df (pd.DataFrame): DataFrame with Volume data
        periods (list): List of periods for volume moving averages

    Returns:
        dict: Dictionary with volume moving averages
    """
    volume_ma_dict = {}
    for period in periods:
        volume_ma_dict[f'Volume_vs_MA_{period}d'] = df['Volume'] / df['Volume'].rolling(window=period).mean()
    return volume_ma_dict

def compute_mfi(df, period=14):
    """
    Compute Money Flow Index (MFI).

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        period (int): Period for MFI calculation

    Returns:
        pd.Series: MFI values
    """
    mfi = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=period)
    return mfi

def compute_adl_zscore(df, rolling_window=20):
    """
    Compute Accumulation/Distribution Line (ADL) z-score.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.Series: ADL values
    """
    adl = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    return (adl - adl.rolling(window=rolling_window).mean()) \
                    / adl.rolling(window=rolling_window).std()

def compute_cmf(df, period=20):
    """
    Compute Chaikin Money Flow (CMF).

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        period (int): Period for CMF calculation

    Returns:
        pd.Series: CMF values
    """
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    cmf = mf_volume.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return cmf

def compute_volume_ratio(df, period=20):
    """
    Compute volume ratio (today's volume vs. average).

    Args:
        df (pd.DataFrame): DataFrame with Volume data
        period (int): Period for volume moving average

    Returns:
        pd.Series: Volume ratio values
    """
    volume_ma = df['Volume'].rolling(window=period).mean()
    volume_ratio = df['Volume'] / volume_ma
    return volume_ratio

def process_stock_data(input_file, output_dir):
    """
    Process a single stock file to generate return data with technical indicators.

    Args:
        input_file (Path): Path to input CSV file
        output_dir (Path): Directory to save processed data
    """
    print(f"Processing {input_file.name}...")

    # Read the stock data
    df = pd.read_csv(input_file)

    # Ensure Date column is sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Compute log returns for price columns
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        df[f'{col}_log_return'] = compute_log_returns(df[col])

    # Keep Volume as is (no log return for volume)
    df['Volume'] = df['Volume']

    # Compute forward returns as target variables
    df['forward_return_7d'] = compute_forward_returns(df['Close'], 7)
    df['forward_return_30d'] = compute_forward_returns(df['Close'], 30)

    # Compute technical indicators
    print(f"  Computing technical indicators...")

    # 1. Moving Averages (5 and 20 days) relative to Close
    ma_dict = compute_moving_averages_relative_to_close(df, periods=[5, 20])
    for ma_name, ma_values in ma_dict.items():
        df[ma_name] = ma_values

    # 2. RSI (14 days)
    df['RSI_14d'] = compute_rsi(df, period=14)

    # 3. MACD from Log Returns (Price-Independent)
    macd_dict = compute_macd_from_log_returns(df, fast=12, slow=26, signal=9)
    for macd_name, macd_values in macd_dict.items():
        df[macd_name] = macd_values

    # 4. Bollinger Bands
    bb_dict = compute_normalized_bollinger_bands(df, period=20, std_dev=2)
    for bb_name, bb_values in bb_dict.items():
        df[bb_name] = bb_values

    # 5. Average True Range (ATR) - 14 days
    df['ATR_relative_14d'] = compute_atr(df, period=14) / df['Close']

    # 6. On-Balance Volume (OBV) z-score
    df['OBV_zscore'] = compute_obv_zscore(df)

    # 7. Volume Weighted Average Price (VWAP)
    df['VWAP_diff'] = np.log(df['Close'] / compute_vwap(df))

    # 8. Volume Moving Averages
    volume_ma_dict = compute_relative_volume_moving_averages(df, periods=[5, 20])
    for volume_ma_name, volume_ma_values in volume_ma_dict.items():
        df[volume_ma_name] = volume_ma_values

    # 9. Money Flow Index (MFI) - 14 days
    df['MFI_14d'] = compute_mfi(df, period=14)

    # 10. Accumulation/Distribution Line (ADL)
    df['ADL_zscore'] = compute_adl_zscore(df)

    # 11. Chaikin Money Flow (CMF) - 20 days
    df['CMF_20d'] = compute_cmf(df, period=20)

    # 12. Volume Ratio (today's volume vs. 20-day average)
    df['Volume_ratio'] = compute_volume_ratio(df, period=20)

    # Select all columns for the dataset
    all_columns = [
        'Date',
        # Log returns
        'Open_log_return', 'High_log_return', 'Low_log_return', 'Close_log_return', 'Volume',
        # Moving Averages
        'Close_vs_MA5_5d', 'Close_vs_MA5_20d',
        # RSI
        'RSI_14d',
        # MACD
        'MACD_line', 'MACD_signal', 'MACD_histogram',
        # Bollinger Bands
        'BB_position',
        # ATR
        'ATR_relative_14d',
        # Volume-based indicators
        'OBV_zscore', 'VWAP_diff',
        'Volume_vs_MA_5d', 'Volume_vs_MA_20d',
        'MFI_14d', 'ADL_zscore',
        'CMF_20d', 'Volume_ratio',
        # Target variables
        'forward_return_7d', 'forward_return_30d'
    ]

    df_features = df[all_columns].copy()

    # Save to output directory
    output_file = output_dir / input_file.name
    df_features.to_csv(output_file, index=False)

    print(f"  Data shape: {df_features.shape}")
    print(f"  Features: {len(all_columns) - 1} (excluding Date)")
    print(f"  Total features: {len(all_columns) - 1}")
    print(f"  Price-based features: 12 (4 log returns + 8 price-based indicators)")
    print(f"  Volume-based features: 8 (OBV, VWAP, Volume_MA_5d, Volume_MA_20d, MFI_14d, ADL, CMF_20d, Volume_ratio)")
    print(f"  Target variables: 2 (forward_return_7d, forward_return_30d)")
    print(f"  Saved to: {output_file}")

    return df_features

def main():
    """
    Main function to process all stock data files and generate technical indicators.
    """
    parser = argparse.ArgumentParser(description='Generate technical indicators for all or a single stock.')
    parser.add_argument('--symbol', '-s', type=str, default=None, help='Process only this symbol (e.g. AAPL)')
    args = parser.parse_args()

    # Define paths
    input_dir = Path('../adjusted_data')
    output_dir = Path('../adjusted_return_ta_data_extended')

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get files to process
    if args.symbol:
        csv_files = [input_dir / f"{args.symbol.upper()}.csv"]
    else:
        csv_files = list(input_dir.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Process each file
    processed_files = []
    total_rows = 0

    for csv_file in csv_files:
        try:
            df_features = process_stock_data(csv_file, output_dir)
            processed_files.append(csv_file.name)
            total_rows += len(df_features)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")

    # Summary statistics
    print("\n" + "="*60)
    print("TECHNICAL INDICATOR GENERATION SUMMARY")
    print("="*60)
    print(f"Files processed: {len(processed_files)}")
    print(f"Total rows: {total_rows:,}")

    # Show sample of processed data
    if processed_files:
        sample_file = output_dir / processed_files[0]
        if sample_file.exists():
            print(f"\nSample of processed data from {processed_files[0]}:")
            df_sample = pd.read_csv(sample_file)
            print(df_sample.head())
            print(f"\nColumns in feature data:")
            for col in df_sample.columns:
                print(f"  - {col}")

    print(f"\nAll extended technical indicator data saved to: {output_dir}")
    print("Features included:")
    print("  - Price-based: 4 log returns + 10 indicators (MA, RSI, MACD, Bollinger Bands, ATR)")
    print("  - Volume-based: 8 indicators (OBV, VWAP, Volume MAs, MFI, ADL, CMF, Volume ratio)")
    print("  - Total: 20 features + 2 target variables")

if __name__ == "__main__":
    main()
