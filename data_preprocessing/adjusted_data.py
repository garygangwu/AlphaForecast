import os
import pandas as pd
from pathlib import Path
import argparse

def process_csv_files(symbol=None):
    # Define paths
    data_dir = Path('../data')
    output_dir = Path('../adjusted_data')

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    # Remove existing CSV files in output directory
    for file in output_dir.glob('*.csv'):
        file.unlink()
        print(f"Removed {file.name}")

    # Get files to process
    if symbol:
        csv_files = [data_dir / f"{symbol}.csv"]
    else:
        csv_files = list(data_dir.glob('*.csv'))

    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Select required columns and make a copy
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_selected = df[required_columns].copy()

            # Calculate adjustment factor
            adjustment_factor = df['adjusted_close'] / df['close']

            # Update price columns using .loc
            df_selected.loc[:, 'open'] = df_selected['open'] * adjustment_factor
            df_selected.loc[:, 'high'] = df_selected['high'] * adjustment_factor
            df_selected.loc[:, 'low'] = df_selected['low'] * adjustment_factor
            df_selected.loc[:, 'close'] = df['adjusted_close']

            # Update volume column - volume should be adjusted inversely to maintain market value
            df_selected.loc[:, 'volume'] = (df_selected['volume'] / adjustment_factor).astype('int64')

            # Create output filename
            output_file = output_dir / csv_file.name

            # Sort by timestamp
            df_selected = df_selected.sort_values(by='timestamp')

            # Rename columns to standard format
            df_selected = df_selected.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Save to new CSV file
            df_selected.to_csv(output_file, index=False)
            print(f"Processed {csv_file.name}")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust stock data for splits. Process all or a single symbol.')
    parser.add_argument('--symbol', '-s', type=str, default=None, help='Process only this symbol (e.g. AAPL)')
    args = parser.parse_args()
    process_csv_files(symbol=args.symbol.upper() if args.symbol else None)
