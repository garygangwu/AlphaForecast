import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import RobustScaler
import json
import argparse

def fit_global_scalers(all_data, feature_columns, output_dir):
    """
    Fit global scalers on all data and save parameters.

    Args:
        all_data (pd.DataFrame): Combined data from all stocks
        feature_columns (list): List of feature columns to normalize
        output_dir (Path): Directory to save scaler parameters

    Returns:
        dict: Dictionary of fitted scalers
    """
    print(f"Fitting global scalers on all data...")

    scalers = {}
    all_scaler_params = {}

    for feature in feature_columns:
        print(f"  Fitting scaler for {feature}...")
        scaler = RobustScaler()

        # Get valid data for this feature
        valid_data = all_data[feature].dropna().values.reshape(-1, 1)

        if len(valid_data) == 0:
            print(f"  Warning: No valid data for {feature}")
            continue

        # Fit scaler
        scaler.fit(valid_data)
        scalers[feature] = scaler

        # Store scaler parameters for combined file
        all_scaler_params[feature] = {
            'center_': scaler.center_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'feature_name': feature
        }

    # Save all scaler parameters in one combined file
    combined_scaler_file = output_dir / "global_scalers_combined.json"
    with open(combined_scaler_file, 'w') as f:
        json.dump(all_scaler_params, f, indent=2)

    print(f"  All scaler parameters saved to: {combined_scaler_file}")

    return scalers

def normalize_features_global(df, feature_columns, scalers):
    """
    Normalize features using pre-fitted global scalers.

    Args:
        df (pd.DataFrame): DataFrame with features
        feature_columns (list): List of feature columns to normalize
        scalers (dict): Dictionary of fitted scalers

    Returns:
        pd.DataFrame: DataFrame with normalized features
    """
    print(f"  Normalizing features using global scalers...")

    normalized_data = df[feature_columns].copy()

    for feature in feature_columns:
        if feature in scalers:
            scaler = scalers[feature]
            # Transform the feature (NaN values will remain NaN)
            normalized_data[feature] = scaler.transform(
                normalized_data[feature].values.reshape(-1, 1)
            ).flatten()

    return normalized_data

def process_stock_normalization(input_file, output_dir, scalers, feature_columns):
    """
    Process a single stock file to apply global normalization.

    Args:
        input_file (Path): Path to input CSV file
        output_dir (Path): Directory to save normalized data
        scalers (dict): Dictionary of fitted scalers
        feature_columns (list): List of feature columns to normalize
    """
    print(f"Normalizing {input_file.name}...")

    # Read the stock data
    df = pd.read_csv(input_file)

    # Apply normalization to technical indicators
    normalized_features = normalize_features_global(df, feature_columns, scalers)

    # Replace original features with normalized ones
    for col in feature_columns:
        df[col] = normalized_features[col]

    # Save to output directory
    output_file = output_dir / input_file.name
    df.to_csv(output_file, index=False)

    print(f"  Data shape: {df.shape}")
    print(f"  Features normalized: {len(feature_columns)}")
    print(f"  Latest date processed: {df['Date'].iloc[-1]}")
    print(f"  Saved to: {output_file}")

    return df

def load_global_scalers_from_file(scaler_file, feature_columns):
    """
    Load global scalers from a combined JSON file.
    """
    with open(scaler_file, 'r') as f:
        scaler_params = json.load(f)
    scalers = {}
    for feature in feature_columns:
        if feature in scaler_params:
            params = scaler_params[feature]
            scaler = RobustScaler()
            scaler.center_ = np.array(params['center_'])
            scaler.scale_ = np.array(params['scale_'])
            scalers[feature] = scaler
    return scalers

def main():
    parser = argparse.ArgumentParser(description='Normalize technical indicator data using global scalers.')
    parser.add_argument('--symbol', '-s', type=str, default=None, help='Process only this symbol (e.g. AAPL)')
    args = parser.parse_args()

    input_dir = Path('../training_data_raw')
    output_dir = Path('../training_data_normalized')
    output_dir.mkdir(exist_ok=True)
    # Remove existing CSV files in output directory
    for file in output_dir.glob('*.csv'):
        file.unlink()
        print(f"Removed {file.name}")

    # Dynamically determine feature columns from the first CSV file
    # Exclude Date, target variables, and one-hot encoded fields from normalization
    exclude_columns = ['Date', 'forward_return_7d', 'forward_return_30d']

    # Get feature columns from the first available CSV file
    csv_files = list(input_dir.glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        print("Please run generate_stock_technical_data.py first to create the technical indicator data.")
        return

    # Read the first file to get column names
    sample_df = pd.read_csv(csv_files[0])
    all_columns = sample_df.columns.tolist()

    # Filter out excluded columns and one-hot encoded candlestick patterns
    feature_columns = []
    one_hot_columns = []

    for col in all_columns:
        if col not in exclude_columns:
            # Check if it's a one-hot encoded candlestick pattern (ends with _Bullish, _Bearish, or _None)
            if col.endswith(('_Bullish', '_Bearish', '_None')):
                one_hot_columns.append(col)
            else:
                feature_columns.append(col)

    print(f"Detected {len(feature_columns)} feature columns to normalize:")
    for col in feature_columns:
        print(f"  - {col}")
    print(f"\nDetected {len(one_hot_columns)} one-hot encoded columns (excluded from normalization):")
    for col in one_hot_columns:
        print(f"  - {col}")
    print(f"\nExcluded columns: {exclude_columns}")
    print(f"Total features: {len(feature_columns) + len(one_hot_columns)}")

    if args.symbol:
        # Single symbol mode: load global scalers and normalize only this file
        symbol = args.symbol.upper()
        csv_file = input_dir / f"{symbol}.csv"
        scaler_file = output_dir / "global_scalers_combined.json"
        if not csv_file.exists():
            print(f"File not found: {csv_file}")
            return
        if not scaler_file.exists():
            print(f"Global scaler file not found: {scaler_file}")
            return

        # Get feature columns from the target file
        target_df = pd.read_csv(csv_file)
        all_columns = target_df.columns.tolist()

        # Filter out excluded columns and one-hot encoded candlestick patterns
        feature_columns = []
        one_hot_columns = []

        for col in all_columns:
            if col not in exclude_columns:
                # Check if it's a one-hot encoded candlestick pattern
                if col.endswith(('_Bullish', '_Bearish', '_None')):
                    one_hot_columns.append(col)
                else:
                    feature_columns.append(col)

        print(f"Normalizing only {csv_file.name} using global scalers from {scaler_file}")
        print(f"Detected {len(feature_columns)} feature columns to normalize")
        print(f"Detected {len(one_hot_columns)} one-hot encoded columns (excluded from normalization)")
        scalers = load_global_scalers_from_file(scaler_file, feature_columns)
        process_stock_normalization(csv_file, output_dir, scalers, feature_columns)
        print(f"Done. Normalized file saved to {output_dir / csv_file.name}")
        return

    # Default: process all files as before
    print(f"Found {len(csv_files)} CSV files to normalize")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Step 1: Collect all data to fit global scalers
    print("Step 1: Collecting all data to fit global scalers...")
    all_data_list = []
    for csv_file in csv_files:
        try:
            df_temp = pd.read_csv(csv_file)
            all_data_list.append(df_temp[feature_columns])
        except Exception as e:
            print(f"Error reading {csv_file.name}: {str(e)}")
    if not all_data_list:
        print("No data collected for scaler fitting")
        return
    all_data = pd.concat(all_data_list, ignore_index=True)
    print(f"Combined data shape: {all_data.shape}")

    # Step 2: Fit global scalers
    scalers = fit_global_scalers(all_data, feature_columns, output_dir)

    # Step 3: Process each file with global scalers
    print("\nStep 2: Processing files with global scalers...")
    processed_files = []
    total_rows = 0
    for csv_file in csv_files:
        try:
            df_features = process_stock_normalization(csv_file, output_dir, scalers, feature_columns)
            processed_files.append(csv_file.name)
            total_rows += len(df_features)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")

    # Summary statistics
    print("\n" + "="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    print(f"Files processed: {len(processed_files)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Features normalized: {len(feature_columns)}")
    print(f"One-hot encoded features (excluded from normalization): {len(one_hot_columns)}")
    print(f"Total features: {len(feature_columns) + len(one_hot_columns)}")
    if processed_files:
        sample_file = output_dir / processed_files[0]
        if sample_file.exists():
            print(f"\nSample of normalized data from {processed_files[0]}:")
            df_sample = pd.read_csv(sample_file)
            print(df_sample.head())
            print(f"\nColumns in normalized data:")
            for col in df_sample.columns:
                print(f"  - {col}")
    print(f"\nAll normalized data saved to: {output_dir}")
    print("Global scaler parameters saved to: global_scalers_combined.json")
    print("\nNote: One-hot encoded candlestick patterns were excluded from normalization")
    print("as they are already in the optimal 0-1 range for deep learning.")

if __name__ == "__main__":
    main()
