#!/bin/bash

# Check if symbol is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <SYMBOL>"
    echo "Example: $0 HOOD"
    exit 1
fi

SYMBOL=$1
SYMBOL_UPPER=$(echo $SYMBOL | tr '[:lower:]' '[:upper:]')

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

# Step 5: Normalize data
echo ""
echo "Step 5/5: Normalizing data..."
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
echo "  - ../adjusted_return_ta_data_extended/$SYMBOL_UPPER.csv (with technical indicators)"
echo "  - ../adjusted_return_ta_data_extended_normalized/$SYMBOL_UPPER.csv (normalized)"
echo ""
echo "The normalized file is ready for model training and prediction!"
echo "=========================================="
