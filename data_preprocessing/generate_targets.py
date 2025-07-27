import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse

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

def process_stock_targets(stock_file, source_dir, output_dir):
    """
    Process a single stock file to add forward return targets.

    Args:
        stock_file (Path): Path to stock CSV file
        source_dir (Path): Directory containing source stock data with Close prices
        output_dir (Path): Output directory for processed data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing {stock_file.name}...")

        # Read the merged stock data
        df = pd.read_csv(stock_file)

        # Ensure Date column is sorted
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Get stock symbol from filename
        stock_symbol = stock_file.stem  # Remove .csv extension

        # Read the source stock data to get actual Close prices
        source_file = source_dir / f"{stock_symbol}.csv"
        if not source_file.exists():
            print(f"  Error: Source file {source_file} not found!")
            return False

        source_df = pd.read_csv(source_file)
        source_df['Date'] = pd.to_datetime(source_df['Date'])
        source_df = source_df.sort_values('Date').reset_index(drop=True)

        # Merge source Close prices with merged data
        source_close = source_df[['Date', 'Close']].copy()
        df_with_close = pd.merge(df, source_close, on='Date', how='left')

        # Check if we have Close prices for all dates
        missing_close = df_with_close['Close'].isnull().sum()
        if missing_close > 0:
            print(f"  Warning: {missing_close} rows have missing Close prices")

        # Compute forward returns as target variables
        df_with_close['forward_return_7d'] = compute_forward_returns(df_with_close['Close'], 7)
        df_with_close['forward_return_30d'] = compute_forward_returns(df_with_close['Close'], 30)

        # Remove the Close column (we don't need it in the final output)
        df_with_close = df_with_close.drop('Close', axis=1)

        # Save to output directory
        output_file = output_dir / stock_file.name
        df_with_close.to_csv(output_file, index=False)

        print(f"  Added forward return targets")
        print(f"  Data shape: {df_with_close.shape}")
        print(f"  Target variables: forward_return_7d, forward_return_30d")

        # Show some statistics about the targets
        print(f"  forward_return_7d stats:")
        print(f"    - Mean: {df_with_close['forward_return_7d'].mean():.6f}")
        print(f"    - Std: {df_with_close['forward_return_7d'].std():.6f}")
        print(f"    - Min: {df_with_close['forward_return_7d'].min():.6f}")
        print(f"    - Max: {df_with_close['forward_return_7d'].max():.6f}")
        print(f"    - Non-null: {df_with_close['forward_return_7d'].count()}")

        print(f"  forward_return_30d stats:")
        print(f"    - Mean: {df_with_close['forward_return_30d'].mean():.6f}")
        print(f"    - Std: {df_with_close['forward_return_30d'].std():.6f}")
        print(f"    - Min: {df_with_close['forward_return_30d'].min():.6f}")
        print(f"    - Max: {df_with_close['forward_return_30d'].max():.6f}")
        print(f"    - Non-null: {df_with_close['forward_return_30d'].count()}")

        return True

    except Exception as e:
        print(f"  Error processing {stock_file.name}: {str(e)}")
        return False

def main():
    """
    Main function to generate forward return targets for all merged stock data.
    """
    parser = argparse.ArgumentParser(description='Generate forward return targets for merged stock data.')
    parser.add_argument('--input_dir', type=str, default='../stock_merged_feature_data',
                       help='Directory containing merged stock data files')
    parser.add_argument('--source_dir', type=str, default='../adjusted_data',
                       help='Directory containing source stock data with Close prices')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Process only this symbol (e.g. AAPL)')
    parser.add_argument('--output_dir', type=str, default='../training_data_raw',
                       help='Directory to save the generated training data')
    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    source_dir = Path(args.source_dir)

    print("="*60)
    print("FORWARD RETURN TARGET GENERATION")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)

    # Check if directories exist
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist!")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get stock files to process
    if args.symbol:
        stock_files = [input_dir / f"{args.symbol.upper()}.csv"]
    else:
        # batch process all stocks
        # Remove existing CSV files in output directory
        for file in output_dir.glob('*.csv'):
            file.unlink()
            print(f"Removed {file.name}")
        stock_files = list(input_dir.glob('*.csv'))

    if not stock_files:
        print(f"No stock files found in {input_dir}")
        return

    print(f"\nFound {len(stock_files)} stock files to process")
    print("="*60)

    # Process each stock file
    processed_files = []
    total_rows = 0

    for stock_file in stock_files:
        success = process_stock_targets(stock_file, source_dir, output_dir)
        if success:
            processed_files.append(stock_file.name)
            # Count rows from the processed file
            df_temp = pd.read_csv(output_dir / stock_file.name)
            total_rows += len(df_temp)

    # Summary
    print("\n" + "="*60)
    print("TARGET GENERATION SUMMARY")
    print("="*60)
    print(f"Files processed: {len(processed_files)}")
    print(f"Total rows: {total_rows:,}")

    if processed_files:
        print(f"\nSample of processed files:")
        for file in processed_files[:5]:  # Show first 5 files
            print(f"  - {file}")
        if len(processed_files) > 5:
            print(f"  ... and {len(processed_files) - 5} more files")

    print(f"\nAll files saved to: {output_dir}")
    print("\nTarget variables added:")
    print("  - forward_return_7d: 7-day forward log return")
    print("  - forward_return_30d: 30-day forward log return")
    print("\nNote: Forward returns are computed from actual Close prices")
    print("from the source data directory.")

if __name__ == "__main__":
    main()
