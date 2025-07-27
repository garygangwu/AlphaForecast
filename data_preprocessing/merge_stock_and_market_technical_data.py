import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse

def load_market_data(market_data_path):
    """
    Load market technical indicators data.

    Args:
        market_data_path (Path): Path to market technical indicators CSV file

    Returns:
        pd.DataFrame: Market technical indicators data
    """
    print(f"Loading market technical data from: {market_data_path}")

    if not market_data_path.exists():
        raise FileNotFoundError(f"Market data file not found: {market_data_path}")

    market_df = pd.read_csv(market_data_path)
    market_df['Date'] = pd.to_datetime(market_df['Date'])

    print(f"  Loaded {len(market_df)} rows of market data")
    print(f"  Date range: {market_df['Date'].min()} to {market_df['Date'].max()}")
    print(f"  Market features: {len(market_df.columns) - 1} (excluding Date)")

    return market_df

def create_relative_measures(stock_df, market_df):
    """
    Create stock relative measures by comparing stock features to market features.

    Args:
        stock_df (pd.DataFrame): Stock technical data
        market_df (pd.DataFrame): Market technical data

    Returns:
        pd.DataFrame: Stock data with relative measures added
    """
    print("  Creating stock relative measures...")

    # Create relative measures
    relative_features = {}

    # 1. Close_log_return - SPY_Close_log_return
    if 'Close_log_return' in stock_df.columns and 'SPY_close_log_return' in market_df.columns:
        relative_features['Close_log_return_vs_SPY'] = stock_df['Close_log_return'] - market_df['SPY_close_log_return']

    # 2. Close_log_return - QQQ_Close_log_return
    if 'Close_log_return' in stock_df.columns and 'QQQ_close_log_return' in market_df.columns:
        relative_features['Close_log_return_vs_QQQ'] = stock_df['Close_log_return'] - market_df['QQQ_close_log_return']

    # 3. RSI_14d - SPY_RSI_14d
    if 'RSI_14d' in stock_df.columns and 'SPY_RSI_14d' in market_df.columns:
        relative_features['RSI_14d_vs_SPY'] = stock_df['RSI_14d'] - market_df['SPY_RSI_14d']

    # 4. Close_vs_MA5_20d - SPY_Close_vs_MA_20d
    if 'Close_vs_MA5_20d' in stock_df.columns and 'SPY_Close_vs_MA_20d' in market_df.columns:
        relative_features['Close_vs_MA5_20d_vs_SPY'] = stock_df['Close_vs_MA5_20d'] - market_df['SPY_Close_vs_MA_20d']

    # Add relative features to stock dataframe
    for feature_name, feature_values in relative_features.items():
        stock_df[feature_name] = feature_values

    print(f"    Added {len(relative_features)} relative measures")
    for feature in relative_features.keys():
        print(f"      - {feature}")

    return stock_df

def merge_stock_with_market_data(stock_file, market_df, output_dir):
    """
    Merge individual stock data with market technical data.

    Args:
        stock_file (Path): Path to stock technical data file
        market_df (pd.DataFrame): Market technical data
        output_dir (Path): Output directory for merged data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing {stock_file.name}...")

        # Load stock data
        stock_df = pd.read_csv(stock_file)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        print(f"  Stock data: {len(stock_df)} rows")
        print(f"  Stock features: {len(stock_df.columns) - 1} (excluding Date)")

        # Merge with market data on Date
        merged_df = pd.merge(
            stock_df,
            market_df,
            on='Date',
            how='left'
        )

        print(f"  After merge: {len(merged_df)} rows")
        print(f"  Total features: {len(merged_df.columns) - 1} (excluding Date)")

        # Create relative measures
        merged_df = create_relative_measures(merged_df, market_df)

        # Save merged data
        output_file = output_dir / stock_file.name
        merged_df.to_csv(output_file, index=False)

        print(f"  Saved to: {output_file}")
        return True

    except Exception as e:
        print(f"  Error processing {stock_file.name}: {str(e)}")
        return False

def main():
    """
    Main function to merge stock and market technical data.
    """
    parser = argparse.ArgumentParser(description='Merge stock technical data with market technical data.')
    parser.add_argument('--stock_data_dir', type=str, default='../stock_technical_data',
                       help='Directory containing stock technical data files')
    parser.add_argument('--market_data_file', type=str, default='../market_data/market_technical_indicators.csv',
                       help='Path to market technical indicators CSV file')
    parser.add_argument('--output_dir', type=str, default='../stock_merged_feature_data',
                       help='Output directory for merged data')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Process only this symbol (e.g. AAPL)')
    args = parser.parse_args()

    # Setup paths
    stock_data_dir = Path(args.stock_data_dir)
    market_data_file = Path(args.market_data_file)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    # Remove existing CSV files in output directory
    for file in output_dir.glob('*.csv'):
        file.unlink()
        print(f"Removed {file.name}")

    print("="*60)
    print("STOCK AND MARKET DATA MERGE")
    print("="*60)
    print(f"Stock data directory: {stock_data_dir}")
    print(f"Market data file: {market_data_file}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Check if directories exist
    if not stock_data_dir.exists():
        print(f"Error: Stock data directory {stock_data_dir} does not exist!")
        return

    if not market_data_file.exists():
        print(f"Error: Market data file {market_data_file} does not exist!")
        return

    # Load market data
    try:
        market_df = load_market_data(market_data_file)
    except Exception as e:
        print(f"Error loading market data: {str(e)}")
        return

    # Get stock files to process
    if args.symbol:
        stock_files = [stock_data_dir / f"{args.symbol.upper()}.csv"]
        if not stock_files[0].exists():
            print(f"Error: Stock file {stock_files[0]} does not exist!")
            return
    else:
        stock_files = list(stock_data_dir.glob('*.csv'))

    if not stock_files:
        print(f"No stock files found in {stock_data_dir}")
        return

    print(f"\nFound {len(stock_files)} stock files to process")
    print("="*60)

    # Process each stock file
    processed_files = []
    total_rows = 0
    total_features = 0

    for stock_file in stock_files:
        success = merge_stock_with_market_data(stock_file, market_df, output_dir)
        if success:
            processed_files.append(stock_file.name)
            # Count rows and features from the saved file
            output_file = output_dir / stock_file.name
            if output_file.exists():
                df_temp = pd.read_csv(output_file)
                total_rows += len(df_temp)
                if total_features == 0:
                    total_features = len(df_temp.columns) - 1  # Exclude Date

    # Summary
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    print(f"Files processed: {len(processed_files)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Total features per stock: {total_features}")

    if processed_files:
        print(f"\nSample of processed files:")
        for file in processed_files[:5]:  # Show first 5 files
            print(f"  - {file}")
        if len(processed_files) > 5:
            print(f"  ... and {len(processed_files) - 5} more files")

    print(f"\nAll merged data saved to: {output_dir}")
    print("\nFeature breakdown:")
    print("  - Original stock technical indicators")
    print("  - Market technical indicators (20 features)")
    print("  - Stock relative measures (4 features):")
    print("    * Close_log_return_vs_SPY")
    print("    * Close_log_return_vs_QQQ")
    print("    * RSI_14d_vs_SPY")
    print("    * Close_vs_MA5_20d_vs_SPY")

if __name__ == "__main__":
    main()
