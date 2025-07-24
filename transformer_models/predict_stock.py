import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.metrics import confusion_matrix

# Import shared model components
from model_common import (
    SEQ_LEN, CLASS_NAMES,
    print_device_info, load_trained_model, load_stock_data, interpret_predictions
)

STOCK_DATA_DIR = '../adjusted_return_ta_data_extended_normalized'
PIC_DIR = 'pics'
Path(PIC_DIR).mkdir(parents=True, exist_ok=True) # Ensure PIC_DIR exists
CLASS_COLORS = {
    'big_down': 'red',
    'down': 'orange',
    'no_change': 'gray',
    'up': 'blue',
    'big_up': 'green'
}

penalty_matrix = np.array([
    [0, 1, 2, 3, 4],
    [1, 0, 1, 2, 3],
    [2, 1, 0, 1, 2],
    [3, 2, 1, 0, 1],
    [4, 3, 2, 1, 0]
])

def custom_f1(y_true, y_pred, penalty_matrix):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    weighted_errors = cm * penalty_matrix
    total_penalty = weighted_errors.sum()

    # Compute max penalty (worst-case scenario: always predict farthest class)
    max_penalty_per_sample = penalty_matrix.max()
    max_total_penalty = len(y_true) * max_penalty_per_sample

    # Compute normalized score (like accuracy)
    score = 1 - (total_penalty / max_total_penalty)

    return score

def load_global_thresholds(cache_file='./global_thresholds.pkl'):
    """
    Load global classification thresholds for interpreting predictions.
    """
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data.get('global_thresholds_7d', {}), cache_data.get('global_thresholds_30d', {})
    except Exception as e:
        print(f"Warning: Could not load global thresholds: {e}")
        print("Classification results may not be accurate without proper thresholds.")
        return {}, {}

def load_stock_data_with_dates(symbol):
    """
    Load return stock data with technical indicators for a given symbol with dates.

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')

    Returns:
        tuple: (stock_data, dates)
    """
    data_dir = Path(STOCK_DATA_DIR)
    file_path = data_dir / f'{symbol}.csv'

    if not file_path.exists():
        raise FileNotFoundError(f"Stock data with technical indicators not found for {symbol}.csv in {data_dir}")

    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)

    # Extract feature data (14 columns: 4 log returns + 10 technical indicators)
    feature_columns = [
        'Open_log_return', 'High_log_return', 'Low_log_return', 'Close_log_return',
        'Close_vs_MA5_5d', 'Close_vs_MA5_20d',
        'RSI_14d',
        'MACD_line', 'MACD_signal', 'MACD_histogram',
        'BB_position',
        'ATR_relative_14d',
        'OBV_zscore', 'VWAP_diff',
        'Volume_vs_MA_5d', 'Volume_vs_MA_20d',
        'MFI_14d', 'ADL_zscore',
        'CMF_20d', 'Volume_ratio',
    ]
    target_columns = ['forward_return_7d', 'forward_return_30d']

    # Combine features and targets
    all_columns = feature_columns + target_columns
    data = df[all_columns].values
    return torch.tensor(data, dtype=torch.float32), df['Date'].values

def load_actual_stock_prices(symbol):
    """
    Load actual stock prices for comparison and conversion.

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')

    Returns:
        tuple: (stock_data, dates)
    """
    data_dir = Path('../adjusted_data')
    file_path = data_dir / f'{symbol}.csv'

    if not file_path.exists():
        raise FileNotFoundError(f"Stock price data not found for {symbol}.csv in {data_dir}")

    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)

    # Extract OHLC data
    data = df[['Open', 'High', 'Low', 'Close']].values
    return torch.tensor(data, dtype=torch.float32), df['Date'].values

def classify_prediction(return_value, global_thresholds):
    """
    Classify a single return prediction using global thresholds.
    """
    if not global_thresholds or return_value is None or np.isnan(return_value):
        return None
    if return_value <= global_thresholds.get('very_low', -float('inf')):
        return "big_down"
    elif return_value <= global_thresholds.get('low', -float('inf')):
        return "down"
    elif return_value < global_thresholds.get('high', float('inf')):
        return "no_change"
    elif return_value < global_thresholds.get('very_high', float('inf')):
        return "up"
    else:
        return "big_up"

def generate_historical_predictions(symbol, model, return_data, return_dates, price_data, price_dates, global_thresholds_7d, global_thresholds_30d):
    """
    Generate predictions for historical data to compare with actual values.
    Only uses the last year of data for plotting.

    Args:
        symbol (str): Stock symbol
        model: Trained multi-task model
        return_data (torch.Tensor): Historical return data with technical indicators
        return_dates (array): Return data date array
        price_data (torch.Tensor): Historical price data
        price_dates (array): Price data date array
        global_thresholds_7d (dict): Global thresholds for 7-day classification
        global_thresholds_30d (dict): Global thresholds for 30-day classification

    Returns:
        dict: Historical predictions and actual values (last year only)
    """
    print("Generating historical predictions for validation (last year)...")

    predictions = {
        'dates': [],
        'actual_return_7d': [],
        'predicted_return_7d': [],
        'actual_return_30d': [],
        'predicted_return_30d': [],
        'predicted_class_7d': [],
        'predicted_class_30d': [],
        'predicted_class_7d_probs': [],
        'predicted_class_30d_probs': [],
        'regression_based_class_7d': [],
        'regression_based_class_30d': [],
        'actual_class_7d': [],
        'actual_class_30d': [],
        'actual_prices': [],
        'actual_price_7d': [],
        'actual_price_30d': [],
        'predicted_price_7d': [],
        'predicted_price_30d': []
    }

    # We need at least SEQ_LEN days to make predictions
    min_required = SEQ_LEN
    if len(return_data) < min_required:
        print(f"Warning: Not enough data for historical validation. Need {min_required} days, got {len(return_data)}")
        return predictions

    # Calculate the start index for last year (approximately 252 trading days)
    days_to_plot = min(252, len(return_data) - 1)
    start_idx = max(SEQ_LEN, len(return_data) - days_to_plot)

    print(f"Plotting predictions for the last {len(return_data) - start_idx} days of data")

    # Generate predictions for each day in the last year
    for i in range(start_idx, len(return_data)):
        # Get sequence for prediction (first 20 columns: 4 log returns + 16 technical indicators)
        sequence = return_data[i-SEQ_LEN:i, :20].unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            reg_7d_pred, reg_30d_pred, class_7d_pred, class_30d_pred = model(sequence)

        # Get actual values from return data
        actual_return_7d = return_data[i, 20]  # forward_return_7d
        actual_return_30d = return_data[i, 21]  # forward_return_30d

        # Interpret classification predictions
        class_7d_interpretation = interpret_predictions(reg_7d_pred, class_7d_pred, "7d")
        class_30d_interpretation = interpret_predictions(reg_30d_pred, class_30d_pred, "30d")

        # Get regression-based classification using global thresholds
        reg_class_7d = classify_prediction(reg_7d_pred.item(), global_thresholds_7d)
        reg_class_30d = classify_prediction(reg_30d_pred.item(), global_thresholds_30d)

        # Find corresponding price data
        return_date = return_dates[i]

        actual_7d_class = classify_prediction(actual_return_7d, global_thresholds_7d)
        actual_30d_class = classify_prediction(actual_return_30d, global_thresholds_30d)

        try:
            price_idx = np.where(price_dates == return_date)[0][0]
            current_price = price_data[price_idx, 3]  # Close price at current date

            # Calculate predicted prices using returns
            # The return represents log return from current day to future day
            # So: future_price = current_price * exp(return)
            predicted_price_7d = current_price * np.exp(reg_7d_pred.item())
            predicted_price_30d = current_price * np.exp(reg_30d_pred.item())


            # Get actual future prices for comparison
            # The actual return is also log return, so apply same formula
            if not torch.isnan(actual_return_7d):
                actual_price_7d = current_price * np.exp(actual_return_7d.item())
            else:
                actual_price_7d = None

            if not torch.isnan(actual_return_30d):
                actual_price_30d = current_price * np.exp(actual_return_30d.item())
            else:
                actual_price_30d = None
        except (IndexError, ValueError):
            # If we can't find corresponding price data, skip this prediction
            continue


        # Store results
        predictions['dates'].append(return_date)
        predictions['actual_return_7d'].append(actual_return_7d.item())
        predictions['predicted_return_7d'].append(reg_7d_pred.item())
        predictions['actual_return_30d'].append(actual_return_30d.item())
        predictions['predicted_return_30d'].append(reg_30d_pred.item())
        predictions['predicted_class_7d'].append(class_7d_interpretation['classification_prediction'][0])
        predictions['predicted_class_30d'].append(class_30d_interpretation['classification_prediction'][0])
        predictions['predicted_class_7d_probs'].append(class_7d_interpretation['classification_probabilities'][0])
        predictions['predicted_class_30d_probs'].append(class_30d_interpretation['classification_probabilities'][0])
        predictions['regression_based_class_7d'].append(reg_class_7d)
        predictions['regression_based_class_30d'].append(reg_class_30d)
        predictions['actual_class_7d'].append(actual_7d_class)
        predictions['actual_class_30d'].append(actual_30d_class)
        predictions['actual_prices'].append(current_price.item())
        predictions['actual_price_7d'].append(actual_price_7d.item() if actual_price_7d is not None else None)
        predictions['actual_price_30d'].append(actual_price_30d.item() if actual_price_30d is not None else None)
        predictions['predicted_price_7d'].append(predicted_price_7d)
        predictions['predicted_price_30d'].append(predicted_price_30d)

    print(f"Generated {len(predictions['dates'])} historical predictions for the last year")

    return predictions

def plot_predictions_vs_actual(predictions, symbol, save_plots=True):
    """
    Create three separate plots comparing multi-task predictions with actual values.

    Args:
        predictions (dict): Historical predictions and actual values
        symbol (str): Stock symbol
        save_plots (bool): Whether to save plots to files
    """
    if not predictions['dates']:
        print("No historical predictions to plot")
        return

    # Convert dates to datetime for better plotting
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in predictions['dates']]

    # For metrics, only use valid (non-NaN) values, but for plotting, keep all dates and values (including NaN/None)
    # Prepare arrays for plotting (may include NaN/None)
    actual_return_7d_arr = np.array(predictions['actual_return_7d'], dtype=np.float64)
    predicted_return_7d_arr = np.array(predictions['predicted_return_7d'], dtype=np.float64)
    actual_return_30d_arr = np.array(predictions['actual_return_30d'], dtype=np.float64)
    predicted_return_30d_arr = np.array(predictions['predicted_return_30d'], dtype=np.float64)

    mask_7d = ~np.isnan(actual_return_7d_arr) & ~np.isnan(predicted_return_7d_arr)
    mask_30d = ~np.isnan(actual_return_30d_arr) & ~np.isnan(predicted_return_30d_arr)

    if np.any(mask_7d):
        mae_7d = np.mean(np.abs(actual_return_7d_arr[mask_7d] - predicted_return_7d_arr[mask_7d]))
        rmse_7d = np.sqrt(np.mean((actual_return_7d_arr[mask_7d] - predicted_return_7d_arr[mask_7d]) ** 2))
    else:
        mae_7d = float('nan')
        rmse_7d = float('nan')

    if np.any(mask_30d):
        mae_30d = np.mean(np.abs(actual_return_30d_arr[mask_30d] - predicted_return_30d_arr[mask_30d]))
        rmse_30d = np.sqrt(np.mean((actual_return_30d_arr[mask_30d] - predicted_return_30d_arr[mask_30d]) ** 2))
    else:
        mae_30d = float('nan')
        rmse_30d = float('nan')

    # ============================================================================
    # PLOT 1: Regression Analysis (2x2 grid)
    # ============================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle(f'{symbol} - Regression Analysis: Predicted vs Actual Returns and Prices', fontsize=16, fontweight='bold')

    # Plot 1.1: 7-day Return Predictions
    axes1[0, 0].plot(dates, actual_return_7d_arr, label='Actual 7-day Return', color='green', linewidth=2)
    axes1[0, 0].plot(dates, predicted_return_7d_arr, label='Predicted 7-day Return', color='orange', linewidth=2, alpha=0.8)
    axes1[0, 0].set_title('7-Day Forward Return: Predicted vs Actual')
    axes1[0, 0].set_ylabel('Return')
    axes1[0, 0].legend()
    axes1[0, 0].grid(True, alpha=0.3)
    axes1[0, 0].text(0.02, 0.98, f'MAE: {mae_7d:.4f}\nRMSE: {rmse_7d:.4f}',
                    transform=axes1[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Plot 1.2: 30-day Return Predictions
    axes1[0, 1].plot(dates, actual_return_30d_arr, label='Actual 30-day Return', color='purple', linewidth=2)
    axes1[0, 1].plot(dates, predicted_return_30d_arr, label='Predicted 30-day Return', color='brown', linewidth=2, alpha=0.8)
    axes1[0, 1].set_title('30-Day Forward Return: Predicted vs Actual')
    axes1[0, 1].set_ylabel('Return')
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)
    axes1[0, 1].text(0.02, 0.98, f'MAE: {mae_30d:.4f}\nRMSE: {rmse_30d:.4f}',
                    transform=axes1[0, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Plot 1.3: 7-day Price Predictions
    actual_price_7d_arr = np.array([
        v if v is not None else np.nan for v in predictions['actual_price_7d']
    ], dtype=np.float64)
    predicted_price_7d_arr = np.array([
        v if v is not None else np.nan for v in predictions['predicted_price_7d']
    ], dtype=np.float64)

    # For metrics, mask out NaN
    mask_price_7d = ~np.isnan(actual_price_7d_arr) & ~np.isnan(predicted_price_7d_arr)
    if np.any(mask_price_7d):
        mse_price_7d = np.mean((actual_price_7d_arr[mask_price_7d] - predicted_price_7d_arr[mask_price_7d]) ** 2)
        rmse_price_7d = np.sqrt(mse_price_7d)
    else:
        mse_price_7d = float('nan')
        rmse_price_7d = float('nan')

    axes1[1, 0].plot(dates, actual_price_7d_arr, label='Actual Price (7-day)', color='blue', linewidth=2)
    axes1[1, 0].plot(dates, predicted_price_7d_arr, label='Predicted Price (7-day)', color='orange', linewidth=2, alpha=0.8)
    axes1[1, 0].set_title('7-Day Price Predictions')
    axes1[1, 0].set_ylabel('Price ($)')
    axes1[1, 0].legend()
    axes1[1, 0].grid(True, alpha=0.3)
    axes1[1, 0].text(0.02, 0.98, f'MSE: {mse_price_7d:.2f}\nRMSE: {rmse_price_7d:.2f}',
                     transform=axes1[1, 0].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Plot 1.4: 30-day Price Predictions
    actual_price_30d_arr = np.array([
        v if v is not None else np.nan for v in predictions['actual_price_30d']
    ], dtype=np.float64)
    predicted_price_30d_arr = np.array([
        v if v is not None else np.nan for v in predictions['predicted_price_30d']
    ], dtype=np.float64)

    mask_price_30d = ~np.isnan(actual_price_30d_arr) & ~np.isnan(predicted_price_30d_arr)
    if np.any(mask_price_30d):
        mse_price_30d = np.mean((actual_price_30d_arr[mask_price_30d] - predicted_price_30d_arr[mask_price_30d]) ** 2)
        rmse_price_30d = np.sqrt(mse_price_30d)
    else:
        mse_price_30d = float('nan')
        rmse_price_30d = float('nan')

    axes1[1, 1].plot(dates, actual_price_30d_arr, label='Actual Price (30-day)', color='purple', linewidth=2, alpha=0.8)
    axes1[1, 1].plot(dates, predicted_price_30d_arr, label='Predicted Price (30-day)', color='brown', linewidth=2, alpha=0.8)
    axes1[1, 1].set_title('30-Day Price Predictions')
    axes1[1, 1].set_ylabel('Price ($)')
    axes1[1, 1].legend()
    axes1[1, 1].grid(True, alpha=0.3)
    axes1[1, 1].text(0.02, 0.98, f'MSE: {mse_price_30d:.2f}\nRMSE: {rmse_price_30d:.2f}',
                     transform=axes1[1, 1].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    # Rotate x-axis labels for better readability
    for ax in axes1.flat:
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save_plots:
        plot_filename1 = f"{PIC_DIR}/{symbol}_regression_analysis.png"
        plt.savefig(plot_filename1, dpi=300, bbox_inches='tight')
        print(f"Regression analysis plot saved to: {plot_filename1}")
    plt.show()

    # ============================================================================
    # PLOT 2: Classification Analysis (2x2 grid)
    # ============================================================================
    from sklearn.metrics import accuracy_score, classification_report

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'{symbol} - Classification Analysis: Predicted vs Actual Classes', fontsize=16, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=class_name) for class_name, color in CLASS_COLORS.items()]

    # --- 7-day metrics ---
    # Filter out None values and align predictions/actuals for metrics only
    y_true_7d = []
    y_pred_7d = []
    for pred, actual in zip(predictions['predicted_class_7d'], predictions['actual_class_7d']):
        if pred is not None and actual is not None:
            y_pred_7d.append(pred)
            y_true_7d.append(actual)
    # Convert to indices for sklearn
    y_true_7d_idx = [CLASS_NAMES.index(c) if c in CLASS_NAMES else -1 for c in y_true_7d]
    y_pred_7d_idx = [CLASS_NAMES.index(c) if c in CLASS_NAMES else -1 for c in y_pred_7d]
    # Remove any -1s (unknown classes)
    valid_7d = [(yt, yp) for yt, yp in zip(y_true_7d_idx, y_pred_7d_idx) if yt != -1 and yp != -1]
    if valid_7d:
        y_true_7d_idx, y_pred_7d_idx = zip(*valid_7d)
        f1_7d = custom_f1(y_true_7d_idx, y_pred_7d_idx, penalty_matrix)
        acc_7d = accuracy_score(y_true_7d_idx, y_pred_7d_idx)
    else:
        f1_7d = float('nan')
        acc_7d = float('nan')

    # --- 30-day metrics ---
    y_true_30d = []
    y_pred_30d = []
    for pred, actual in zip(predictions['predicted_class_30d'], predictions['actual_class_30d']):
        if pred is not None and actual is not None:
            y_pred_30d.append(pred)
            y_true_30d.append(actual)
    y_true_30d_idx = [CLASS_NAMES.index(c) if c in CLASS_NAMES else -1 for c in y_true_30d]
    y_pred_30d_idx = [CLASS_NAMES.index(c) if c in CLASS_NAMES else -1 for c in y_pred_30d]
    valid_30d = [(yt, yp) for yt, yp in zip(y_true_30d_idx, y_pred_30d_idx) if yt != -1 and yp != -1]
    if valid_30d:
        y_true_30d_idx, y_pred_30d_idx = zip(*valid_30d)
        f1_30d = custom_f1(y_true_30d_idx, y_pred_30d_idx, penalty_matrix)
        acc_30d = accuracy_score(y_true_30d_idx, y_pred_30d_idx)
    else:
        f1_30d = float('nan')
        acc_30d = float('nan')

    # Plot 2.1: 7-day Classification Predictions vs Actual Returns
    # Plot all points, including those with None class_pred
    last_valid_actual_return = None
    for i, (date, class_pred, actual_return) in enumerate(zip(dates, predictions['predicted_class_7d'], predictions['actual_return_7d'])):
        if np.isnan(actual_return):
            actual_return = last_valid_actual_return
        color = CLASS_COLORS.get(class_pred, 'black')
        axes2[0, 0].scatter(date, actual_return, c=color, alpha=0.7, s=30)
        last_valid_actual_return = actual_return

    axes2[0, 0].set_title('7-Day Classification Predictions vs Actual Returns')
    axes2[0, 0].set_ylabel('Actual Return')
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].legend(handles=legend_elements, loc='upper right')
    axes2[0, 0].text(0.02, 0.98, f'F1: {f1_7d:.2f}\nAcc: {acc_7d:.2f}',
                     transform=axes2[0, 0].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Plot 2.2: 30-day Classification Predictions vs Actual Returns
    for i, (date, class_pred, actual_return) in enumerate(zip(dates, predictions['predicted_class_30d'], predictions['actual_return_30d'])):
        if np.isnan(actual_return):
            actual_return = last_valid_actual_return
        color = CLASS_COLORS.get(class_pred, 'black')
        axes2[0, 1].scatter(date, actual_return, c=color, alpha=0.7, s=30)
        last_valid_actual_return = actual_return

    axes2[0, 1].set_title('30-Day Classification Predictions vs Actual Returns')
    axes2[0, 1].set_ylabel('Actual Return')
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].legend(handles=legend_elements, loc='upper right')
    axes2[0, 1].text(0.02, 0.98, f'F1: {f1_30d:.2f}\nAcc: {acc_30d:.2f}',
                     transform=axes2[0, 1].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    # Plot 2.3: 7-day Actual Classes vs Actual Returns
    for i, (date, actual_class, actual_return) in enumerate(zip(dates, predictions['actual_class_7d'], predictions['actual_return_7d'])):
        if not np.isnan(actual_return):
            color = CLASS_COLORS.get(actual_class, 'black')
            axes2[1, 0].scatter(date, actual_return, c=color, alpha=0.7, s=30)
        else:
            axes2[1, 0].scatter(date, 0, c='white', alpha=0.7, s=30)

    axes2[1, 0].set_title('7-Day Actual Classes vs Actual Returns')
    axes2[1, 0].set_ylabel('Actual Return')
    axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].legend(handles=legend_elements, loc='upper right')

    # Plot 2.4: 30-day Actual Classes vs Actual Returns
    for i, (date, actual_class, actual_return) in enumerate(zip(dates, predictions['actual_class_30d'], predictions['actual_return_30d'])):
        if not np.isnan(actual_return):
            color = CLASS_COLORS.get(actual_class, 'black')
            axes2[1, 1].scatter(date, actual_return, c=color, alpha=0.7, s=30)
        else:
            axes2[1, 1].scatter(date, 0, c='white', alpha=0.7, s=30)

    axes2[1, 1].set_title('30-Day Actual Classes vs Actual Returns')
    axes2[1, 1].set_ylabel('Actual Return')
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].legend(handles=legend_elements, loc='upper right')

    # Rotate x-axis labels for better readability
    for ax in axes2.flat:
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save_plots:
        plot_filename2 = f"{PIC_DIR}/{symbol}_classification_analysis.png"
        plt.savefig(plot_filename2, dpi=300, bbox_inches='tight')
        print(f"Classification analysis plot saved to: {plot_filename2}")
    plt.show()

    # ============================================================================
    # PLOT 3: Consistency Report (2x1 grid)
    # ============================================================================
    fig3, axes3 = plt.subplots(2, 1, figsize=(16, 10))
    fig3.suptitle(f'{symbol} - Consistency Report: Regression-based vs Neural Network Classification', fontsize=16, fontweight='bold')

    # Convert class names to numeric indices
    reg_class_7d_numeric = [CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 2 for cls in predictions['regression_based_class_7d']]
    nn_class_7d_numeric = [CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 2 for cls in predictions['predicted_class_7d']]
    reg_class_30d_numeric = [CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 2 for cls in predictions['regression_based_class_30d']]
    nn_class_30d_numeric = [CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 2 for cls in predictions['predicted_class_30d']]

    # Plot 3.1: 7-day Classification Consistency
    axes3[0].plot(dates, reg_class_7d_numeric, label='Regression-based Classification', color='blue', alpha=0.7, linewidth=2)
    axes3[0].plot(dates, nn_class_7d_numeric, label='Neural Network Classification', color='red', alpha=0.7, linewidth=2)
    axes3[0].set_title('7-Day Classification Consistency: Regression vs Neural Network')
    axes3[0].set_ylabel('Class Index')
    axes3[0].set_yticks(range(5))
    axes3[0].set_yticklabels(CLASS_NAMES, rotation=45)
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    # Plot 3.2: 30-day Classification Consistency
    axes3[1].plot(dates, reg_class_30d_numeric, label='Regression-based Classification', color='blue', alpha=0.7, linewidth=2)
    axes3[1].plot(dates, nn_class_30d_numeric, label='Neural Network Classification', color='red', alpha=0.7, linewidth=2)
    axes3[1].set_title('30-Day Classification Consistency: Regression vs Neural Network')
    axes3[1].set_ylabel('Class Index')
    axes3[1].set_xlabel('Date')
    axes3[1].set_yticks(range(5))
    axes3[1].set_yticklabels(CLASS_NAMES, rotation=45)
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    for ax in axes3:
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save_plots:
        plot_filename3 = f"{PIC_DIR}/{symbol}_consistency_report.png"
        plt.savefig(plot_filename3, dpi=300, bbox_inches='tight')
        print(f"Consistency report plot saved to: {plot_filename3}")
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("MULTI-TASK MODEL PERFORMANCE SUMMARY")
    print("="*60)

    print(f"\n7-Day Return Predictions (Regression):")
    print(f"  MAE:  {mae_7d:.4f}")
    print(f"  RMSE: {rmse_7d:.4f}")

    print(f"\n30-Day Return Predictions (Regression):")
    print(f"  MAE:  {mae_30d:.4f}")
    print(f"  RMSE: {rmse_30d:.4f}")

    # Classification accuracy
    class_agreement_7d = sum(1 for r, n in zip(predictions['regression_based_class_7d'], predictions['predicted_class_7d']) if r == n)
    class_agreement_30d = sum(1 for r, n in zip(predictions['regression_based_class_30d'], predictions['predicted_class_30d']) if r == n)

    print(f"\nClassification Consistency:")
    print(f"  7-day: {class_agreement_7d}/{len(predictions['dates'])} ({class_agreement_7d/len(predictions['dates'])*100:.1f}%) agreement between regression and NN classification")
    print(f"  30-day: {class_agreement_30d}/{len(predictions['dates'])} ({class_agreement_30d/len(predictions['dates'])*100:.1f}%) agreement between regression and NN classification")

def predict_stock(symbol, model_path="best_model.pth", plot_historical=True):
    """
    Predict forward returns for a given symbol using multi-task model.

    Args:
        symbol (str): Stock symbol
        model_path (str): Path to the trained model
        plot_historical (bool): Whether to generate historical validation plots

    Returns:
        dict: Predictions with return values, classifications, and converted prices
    """
    print(f"Loading stock data with technical indicators for {symbol}...")

    # Load return data with technical indicators
    return_data, return_dates = load_stock_data_with_dates(symbol)

    # Load actual price data for comparison
    price_data, price_dates = load_actual_stock_prices(symbol)

    # Load global thresholds for classification interpretation
    global_thresholds_7d, global_thresholds_30d = load_global_thresholds()

    print(f"Loaded {len(return_data)} days of return data with technical indicators")
    print(f"Date range: {return_dates[0]} to {return_dates[-1]}")

    # Check if we have enough data
    if len(return_data) < SEQ_LEN:
        raise ValueError(f"Not enough data for {symbol}. Need at least {SEQ_LEN} days, got {len(return_data)}")

    # Load model
    print(f"Loading multi-task model from {model_path}...")
    model = load_trained_model(model_path)

    # Get the most recent sequence (first 20 columns: 4 log returns + 16 technical indicators)
    from model_common import device
    latest_sequence = return_data[-SEQ_LEN:, :20].to(device).unsqueeze(0)  # Add batch dimension

    print(f"Using latest {SEQ_LEN} days for prediction")
    print(f"Latest date in sequence: {return_dates[-1]}")

    # Get current price information
    current_price_idx = np.where(price_dates == return_dates[-1])[0][0]
    current_price = price_data[current_price_idx, 3]  # Close price

    print(f"{return_dates[-1]} stock info:")
    print(f"  Close Price: ${current_price:.2f}")

    # Make prediction
    print("Making multi-task prediction...")
    with torch.no_grad():
        reg_7d_pred, reg_30d_pred, class_7d_pred, class_30d_pred = model(latest_sequence)

    # Convert to numpy for easier processing
    reg_7d_val = reg_7d_pred.cpu().numpy().item()
    reg_30d_val = reg_30d_pred.cpu().numpy().item()

    # Interpret classification predictions
    class_7d_interpretation = interpret_predictions(reg_7d_pred, class_7d_pred, "7d")
    class_30d_interpretation = interpret_predictions(reg_30d_pred, class_30d_pred, "30d")

    # Get regression-based classification using global thresholds
    reg_class_7d = classify_prediction(reg_7d_val, global_thresholds_7d)
    reg_class_30d = classify_prediction(reg_30d_val, global_thresholds_30d)

    # Convert returns to price predictions
    # The return represents log return from current day to future day
    # So: future_price = current_price * exp(return)
    predicted_price_7d = current_price * np.exp(reg_7d_val)
    predicted_price_30d = current_price * np.exp(reg_30d_val)

    # Generate historical predictions for validation if requested
    historical_predictions = None
    if plot_historical:
        historical_predictions = generate_historical_predictions(
            symbol, model, return_data, return_dates, price_data, price_dates,
            global_thresholds_7d, global_thresholds_30d
        )
        plot_predictions_vs_actual(historical_predictions, symbol)

    return {
        'symbol': symbol,
        'current_price': current_price,
        'prediction_date': return_dates[-1],
        'regression_7d_prediction': reg_7d_val,
        'regression_30d_prediction': reg_30d_val,
        'classification_7d_prediction': class_7d_interpretation['classification_prediction'][0],
        'classification_30d_prediction': class_30d_interpretation['classification_prediction'][0],
        'classification_7d_probabilities': class_7d_interpretation['classification_probabilities'][0],
        'classification_30d_probabilities': class_30d_interpretation['classification_probabilities'][0],
        'classification_7d_confidence': class_7d_interpretation['confidence_scores'][0],
        'classification_30d_confidence': class_30d_interpretation['confidence_scores'][0],
        'regression_based_class_7d': reg_class_7d,
        'regression_based_class_30d': reg_class_30d,
        'predicted_price_7d': predicted_price_7d,
        'predicted_price_30d': predicted_price_30d,
        'global_thresholds_7d': global_thresholds_7d,
        'global_thresholds_30d': global_thresholds_30d,
        'historical_predictions': historical_predictions
    }

def print_predictions(predictions, concise=False):
    """
    Print multi-task predictions in a formatted way.

    Args:
        predictions (dict): Prediction results
        concise (bool): If True, print only 2 lines of concise output
    """
    if concise:
        # Concise output format
        symbol = predictions['symbol'].upper()
        reg_7d = predictions['regression_7d_prediction']
        reg_30d = predictions['regression_30d_prediction']
        class_7d = predictions['classification_7d_prediction'].upper()
        class_30d = predictions['classification_30d_prediction'].upper()
        conf_7d = predictions['classification_7d_confidence']
        conf_30d = predictions['classification_30d_confidence']
        price_7d = predictions['predicted_price_7d']
        price_30d = predictions['predicted_price_30d']

        # Determine icons based on regression direction
        icon_7d = "📈" if reg_7d > 0 else "📉"
        icon_30d = "📈" if reg_30d > 0 else "📉"

        print(f"{symbol} {icon_7d} 7-day predicted price:  {reg_7d*100:+.2f}% ${price_7d:.2f} | {icon_7d} Classification: {class_7d} (confidence: {conf_7d:.3f})")
        print(f"{symbol} {icon_30d} 30-day predicted price: {reg_30d*100:+.2f}% ${price_30d:.2f} | {icon_30d} Classification: {class_30d} (confidence: {conf_30d:.3f})")
        return

    # Full detailed output (original format)
    print("\n" + "="*70)
    print(f"MULTI-TASK STOCK PREDICTION RESULTS FOR {predictions['symbol'].upper()}")
    print("="*70)
    print(f"Based on data up to: {predictions['prediction_date']}")
    print(f"Current Price: ${predictions['current_price']:.2f}")
    print()

    print("REGRESSION PREDICTIONS:")
    print("-" * 30)
    print(f"7-day forward return:  {predictions['regression_7d_prediction']:.4f} ({predictions['regression_7d_prediction']*100:.2f}%)")
    print(f"30-day forward return: {predictions['regression_30d_prediction']:.4f} ({predictions['regression_30d_prediction']*100:.2f}%)")
    print()

    print("CLASSIFICATION PREDICTIONS:")
    print("-" * 30)
    print(f"7-day direction:  {predictions['classification_7d_prediction'].upper()} (confidence: {predictions['classification_7d_confidence']:.3f})")
    print(f"30-day direction: {predictions['classification_30d_prediction'].upper()} (confidence: {predictions['classification_30d_confidence']:.3f})")
    print()

    # Show detailed class probabilities
    print("DETAILED CLASS PROBABILITIES:")
    print("-" * 30)
    print("7-day probabilities:")
    for class_name, prob in zip(CLASS_NAMES, predictions['classification_7d_probabilities']):
        print(f"  {class_name}: {prob:.3f}")
    print("30-day probabilities:")
    for class_name, prob in zip(CLASS_NAMES, predictions['classification_30d_probabilities']):
        print(f"  {class_name}: {prob:.3f}")
    print()

    print("REGRESSION-BASED CLASSIFICATION:")
    print("-" * 30)
    print(f"7-day (from global thresholds):  {predictions['regression_based_class_7d'].upper()}")
    print(f"30-day (from global thresholds): {predictions['regression_based_class_30d'].upper()}")
    print()

    print("PRICE PREDICTIONS (Converted from Returns):")
    print("-" * 30)
    print(f"7-day predicted price:  ${predictions['predicted_price_7d']:.2f}")
    print(f"30-day predicted price: ${predictions['predicted_price_30d']:.2f}")
    print()

    # Calculate some insights
    reg_7d = predictions['regression_7d_prediction']
    reg_30d = predictions['regression_30d_prediction']
    class_7d = predictions['classification_7d_prediction']
    class_30d = predictions['classification_30d_prediction']

    print("PREDICTION INSIGHTS:")
    print("-" * 30)

    # 7-day outlook
    reg_icon = "📈" if reg_7d > 0 else "📉"
    class_icon = "📈" if class_7d in ['up', 'big_up'] else "📉" if class_7d in ['down', 'big_down'] else "➡️"
    consistency_7d = "✅" if (reg_7d > 0 and class_7d in ['up', 'big_up']) or (reg_7d < 0 and class_7d in ['down', 'big_down']) or (abs(reg_7d) < 0.01 and class_7d == 'no_change') else "⚠️"

    print(f"{reg_icon} 7-day regression: {reg_7d*100:+.2f}% | {class_icon} Classification: {class_7d.upper()} {consistency_7d}")

    # 30-day outlook
    reg_icon = "📈" if reg_30d > 0 else "📉"
    class_icon = "📈" if class_30d in ['up', 'big_up'] else "📉" if class_30d in ['down', 'big_down'] else "➡️"
    consistency_30d = "✅" if (reg_30d > 0 and class_30d in ['up', 'big_up']) or (reg_30d < 0 and class_30d in ['down', 'big_down']) or (abs(reg_30d) < 0.01 and class_30d == 'no_change') else "⚠️"

    print(f"{reg_icon} 30-day regression: {reg_30d*100:+.2f}% | {class_icon} Classification: {class_30d.upper()} {consistency_30d}")

    # Overall consistency
    overall_consistency = "✅ CONSISTENT" if consistency_7d == "✅" and consistency_30d == "✅" else "⚠️ MIXED SIGNALS"
    print(f"\nOverall Model Consistency: {overall_consistency}")

    if predictions['global_thresholds_7d']:
        print(f"\nGlobal Classification Thresholds (7d): {predictions['global_thresholds_7d']}")
    if predictions['global_thresholds_30d']:
        print(f"Global Classification Thresholds (30d): {predictions['global_thresholds_30d']}")

    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Predict stock returns using trained multi-task transformer model')
    parser.add_argument('symbol', type=str, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating historical validation plots')
    parser.add_argument('--concise', action='store_true',
                       help='Print only concise 2-line output')

    args = parser.parse_args()

    try:
        # Make prediction
        # If concise is True, force plot_historical to True (i.e., no_plots to False)
        plot_historical = not args.no_plots
        if args.concise:
            plot_historical = False
        if not args.concise:
            # Print device info
            print_device_info()
        if args.concise:
            # Suppress all prints in predict_stock by temporarily redirecting stdout
            import sys
            import io
            _stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                predictions = predict_stock(args.symbol, args.model, plot_historical=plot_historical)
            finally:
                sys.stdout = _stdout
        else:
            predictions = predict_stock(args.symbol, args.model, plot_historical=plot_historical)

        # Print results
        print_predictions(predictions, concise=args.concise)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Run the data preprocessing pipeline to create technical indicator data")
        print("2. Run general_model_corrected.py to train the multi-task model")
        print("3. The stock symbol exists in the adjusted_return_ta_data_normalized folder")
        print("4. Global thresholds file (global_thresholds.pkl) exists")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
