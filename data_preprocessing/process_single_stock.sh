#!/bin/bash

# Check if symbol is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <SYMBOL> [--no-confirm]"
    echo "Example: $0 HOOD"
    echo "Example: $0 HOOD --no-confirm"
    exit 1
fi

SYMBOL=$1
SYMBOL_UPPER=$(echo $SYMBOL | tr '[:lower:]' '[:upper:]')

# Check for --no-confirm flag
NO_CONFIRM=false
if [ "$2" = "--no-confirm" ]; then
    NO_CONFIRM=true
fi

echo "=========================================="
echo "please make sure you have downloaded and processed the market data"
echo "by running the following command:"
echo "python download_market_data.py"
echo "python process_market_data.py"
echo "=========================================="

# Ask for confirmation (unless --no-confirm flag is used)
if [ "$NO_CONFIRM" = false ]; then
    echo "This will process $SYMBOL_UPPER through the entire pipeline:"
    echo "1. Clean existing data"
    echo "2. Download new stock data"
    echo "3. Adjust for splits"
    echo "4. Generate technical indicators"
    echo ""
    read -p "Do you want to continue? (y/N) " confirm

    if [[ $confirm != [yY] ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "Proceeding without confirmation (--no-confirm flag used)..."
fi

echo "=========================================="
echo "STOCK DATA PREPROCESSING PIPELINE"
echo "=========================================="
echo "Processing symbol: $SYMBOL_UPPER"
echo "=========================================="

# Step 1: Remove existing data file
echo ""
echo "Step 1/4: Cleaning existing data..."
echo "  Removing ../data/$SYMBOL_UPPER.csv"
rm -f ../data/$SYMBOL_UPPER.csv
echo "  ✓ Cleaned existing data file"

# Step 2: Download stock data
echo ""
echo "Step 2/4: Downloading stock data..."
echo "  Downloading $SYMBOL_UPPER from Alpha Vantage..."
python download_stock_data.py --symbol $SYMBOL_UPPER
if [ $? -eq 0 ]; then
    echo "  ✓ Successfully downloaded $SYMBOL_UPPER.csv"
else
    echo "  ✗ Failed to download $SYMBOL_UPPER.csv"
    exit 1
fi

# Step 3: Adjust data for splits
echo ""
echo "Step 3/4: Adjusting data for stock splits..."
echo "  Processing $SYMBOL_UPPER.csv for splits and volume adjustments..."
python adjusted_data.py --symbol $SYMBOL_UPPER
if [ $? -eq 0 ]; then
    echo "  ✓ Successfully adjusted $SYMBOL_UPPER.csv"
else
    echo "  ✗ Failed to adjust $SYMBOL_UPPER.csv"
    exit 1
fi

# Step 4: Generate technical indicators
echo ""
echo "Step 4/4: Generating technical indicators..."
echo "  Computing 20 technical indicators for $SYMBOL_UPPER..."
python generate_stock_technical_data.py --symbol $SYMBOL_UPPER
if [ $? -eq 0 ]; then
    echo "  ✓ Successfully generated technical indicators"
else
    echo "  ✗ Failed to generate technical indicators"
    exit 1
fi

# Step 5: Merge stock and market technical data
echo ""
echo "Step 5/5: Merging stock and market technical data..."
python merge_stock_and_market_technical_data.py --symbol $SYMBOL_UPPER
if [ $? -eq 0 ]; then
    echo "  ✓ Successfully merged stock and market technical data"
else
    echo "  ✗ Failed to merge stock and market technical data"
    exit 1
fi

# Step 6: Normalize data
echo ""
echo "Step 6/6: Normalizing data..."
echo "  Applying global normalization to $SYMBOL_UPPER..."
python normalize_technical_data.py --symbol $SYMBOL_UPPER
if [ $? -eq 0 ]; then
    echo "  ✓ Successfully normalized $SYMBOL_UPPER"
else
    echo "  ✗ Failed to normalize $SYMBOL_UPPER"
    exit 1
fi

echo ""
echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Symbol: $SYMBOL_UPPER"
echo ""
echo "Output files created:"
echo "  - ../data/$SYMBOL_UPPER.csv (raw data)"
echo "  - ../adjusted_data/$SYMBOL_UPPER.csv (split-adjusted)"
echo "  - ../stock_technical_data/$SYMBOL_UPPER.csv (with technical indicators)"
echo "  - ../stock_merged_feature_data/$SYMBOL_UPPER.csv (merged with market data)"
echo "  - ../training_data_raw/$SYMBOL_UPPER.csv (final training data with targets)"
echo "  - ../training_data_normalized/$SYMBOL_UPPER.csv (normalized)"
echo ""
echo "The normalized file is ready for model training and prediction!"
echo "=========================================="
