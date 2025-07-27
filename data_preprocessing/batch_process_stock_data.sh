#!/bin/bash

# Batch Process Stock Data Pipeline
# This script runs the complete data preprocessing pipeline for stock prediction

set -e  # Exit on any error

echo "============================================================"
echo "BATCH PROCESS STOCK DATA PIPELINE"
echo "============================================================"
echo "Starting complete data preprocessing pipeline..."
echo ""

# Function to print step header
print_step() {
    echo ""
    echo "============================================================"
    echo "STEP $1: $2"
    echo "============================================================"
    echo ""
}

# Function to check if previous step was successful
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ Step completed successfully"
    else
        echo "❌ Step failed with exit code $?"
        exit 1
    fi
}

# Step 1: Clean existing data
print_step "1" "Cleaning existing data files"
echo "Removing existing CSV files from ../data/ directory..."
rm -f ../data/*.csv
check_success

# Step 2: Download stock data
print_step "2" "Downloading stock data"
echo "Running download_stock_data.py..."
python download_stock_data.py
check_success

# Step 3: Download market data
print_step "3" "Downloading market data"
echo "Running download_market_data.py..."
python download_market_data.py --download all
check_success

# Step 4: Generate market technical indicators
print_step "4" "Generating market technical indicators"
echo "Running generate_market_technical_data.py..."
python generate_market_technical_data.py
check_success

# Step 5: Process adjusted data
print_step "5" "Processing adjusted data"
echo "Running adjusted_data.py..."
python adjusted_data.py
check_success

# Step 6: Generate stock technical indicators
print_step "6" "Generating stock technical indicators"
echo "Running generate_stock_technical_data.py..."
python generate_stock_technical_data.py
check_success

# Step 7: Merge stock and market technical data
print_step "7" "Merging stock and market technical data"
echo "Running merge_stock_and_market_technical_data.py..."
python merge_stock_and_market_technical_data.py
check_success

# Step 8: Normalize technical data
print_step "8" "Normalizing technical data"
echo "Running normalize_technical_data.py..."
python normalize_technical_data.py
check_success

# Step 9: Generate targets
print_step "9" "Generating forward return targets"
echo "Running generate_targets.py..."
python generate_targets.py
check_success

echo ""
echo "============================================================"
echo "🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉"
echo "============================================================"
echo ""
echo "Data processing pipeline has been completed. Here's what was generated:"
echo ""
echo "📁 Directory Structure:"
echo "  ../data/                        - Raw downloaded stock data"
echo "  ../market_data/                 - Market data"
echo "  ../adjusted_data/               - Adjusted stock data"
echo "  ../stock_technical_data/        - Stock technical indicators"
echo "  ../stock_merged_feature_data/   - Merged stock and market features"
echo "  ../training_data_raw/           - Final training data with targets"
echo "  ../training_data_normalized/    - Normalized feature data"
echo ""
echo "📊 Data Summary:"
echo "  - Stock technical indicators: 64 features"
echo "  - Market technical indicators: 20 features"
echo "  - Stock relative measures: 4 features"
echo "  - Target variables: 2 (forward_return_7d, forward_return_30d)"
echo "  - Total features per stock: ~90"
echo ""
echo "🚀 Ready for model training!"
echo "============================================================"
