import os
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import random
import argparse
import numpy as np
import pickle
import hashlib
from datetime import datetime

# Import shared model components
from model_common import (
    SEQ_LEN, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES, device,
    print_device_info, load_stock_data, create_model, save_model, load_trained_model,
    consistency_loss, interpret_predictions,
    classify_single_return, calculate_global_thresholds
)

INPUT_DATA_DIR = '../adjusted_return_ta_data_extended_normalized'

# Print device info
print_device_info()

def get_data_hash():
    """
    Generate a hash of the data directory to detect changes.
    This ensures we regenerate sequences if the data has changed.
    """
    csv_dir = Path(INPUT_DATA_DIR)
    csv_files = sorted(list(csv_dir.glob('*.csv')))

    # Create hash based on file names, sizes
    hash_content = []
    for filepath in csv_files:
        stat = filepath.stat()
        hash_content.append(f"{filepath.name}:{stat.st_size}")

    # Create MD5 hash
    hash_string = '|'.join(hash_content)
    return hashlib.md5(hash_string.encode()).hexdigest()

def save_global_thresholds_to_cache(global_thresholds_7d, global_thresholds_30d, cache_file):
    """
    Save sequences and global thresholds to a cache file with metadata.
    """
    cache_data = {
        'global_thresholds_7d': global_thresholds_7d,
        'global_thresholds_30d': global_thresholds_30d,
        'data_hash': get_data_hash(),
        'created_at': datetime.now().isoformat(),
        'version': '2.0'
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Saved global thresholds to local file: {cache_file}")

def load_global_thresholds_from_cache(cache_file):
    """
    Load global thresholds from cache file if valid.
    Returns None if cache is invalid or doesn't exist.
    """
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} does not exist.")
        return None

    try:
        print(f"Loading global thresholds from cache file: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        # Validate cache
        current_hash = get_data_hash()
        if cache_data.get('data_hash') != current_hash:
            print("Cache is invalid: data has changed since cache was created.")
            return None

        # Check if this is the corrected version
        if cache_data.get('version', '').startswith('2.0'):
            global_thresholds_7d = cache_data.get('global_thresholds_7d', {})
            global_thresholds_30d = cache_data.get('global_thresholds_30d', {})

            print(f"Cache loaded successfully!")
            print(f"  Created at: {cache_data.get('created_at', 'unknown')}")
            print(f"  Cache version: {cache_data.get('version', 'unknown')}")
            print(f"  Global 7d thresholds: {global_thresholds_7d}")
            print(f"  Global 30d thresholds: {global_thresholds_30d}")

            return global_thresholds_7d, global_thresholds_30d
        else:
            print("Cache is from old version with incorrect classification. Will regenerate.")
            return None

    except Exception as e:
        print(f"Error loading cache file: {e}")
        return None

def ensure_sequences_on_device(sequences):
    """
    Ensure all tensors in sequences are on the correct device.

    Args:
        sequences: List of sequences with tensors

    Returns:
        List of sequences with tensors moved to the correct device
    """
    print(f"Moving {len(sequences)} sequences to device: {device}")
    device_sequences = []

    for seq_x, return_7d, return_30d, class_7d, class_30d in sequences:
        # Move all tensors to the correct device
        seq_x = seq_x.to(device)
        return_7d = return_7d.to(device) if isinstance(return_7d, torch.Tensor) else torch.tensor(return_7d, device=device)
        return_30d = return_30d.to(device) if isinstance(return_30d, torch.Tensor) else torch.tensor(return_30d, device=device)
        class_7d = class_7d.to(device) if isinstance(class_7d, torch.Tensor) else torch.tensor(class_7d, device=device)
        class_30d = class_30d.to(device) if isinstance(class_30d, torch.Tensor) else torch.tensor(class_30d, device=device)

        device_sequences.append((seq_x, return_7d, return_30d, class_7d, class_30d))

    return device_sequences

def load_all_sequences_corrected(use_cache=True, cache_file='global_thresholds.pkl'):
    """
    CORRECTED VERSION: Load all sequences using GLOBAL classification thresholds.

    This fixes the fundamental issue where classification was based on individual
    return percentiles instead of global distribution statistics.

    Args:
        use_cache: Whether to use caching
        cache_file: Path to cache file

    Returns:
        Tuple of (sequences, global_thresholds_7d, global_thresholds_30d)
    """
    # STEP 1: Collect ALL returns from ALL stocks to calculate GLOBAL percentiles
    print("=" * 60)
    print("STEP 1: Collecting ALL returns for GLOBAL percentile calculation...")
    print("=" * 60)

    # Try to load from cache first
    global_thresholds_7d, global_thresholds_30d = None, None
    if use_cache:
        cached_result = load_global_thresholds_from_cache(cache_file)
        if cached_result is not None:
            global_thresholds_7d, global_thresholds_30d = cached_result

    csv_dir = Path(INPUT_DATA_DIR)
    csv_files = list(csv_dir.glob('*.csv'))
    if global_thresholds_7d is None or global_thresholds_30d is None:
        # If cache loading failed, generate sequences from scratch with CORRECT classification
        print("Generating sequences from scratch with CORRECTED global classification...")

        all_returns_7d = []
        all_returns_30d = []

        for filepath in csv_files:
            stock_data = load_stock_data(filepath)
            returns_7d = stock_data[SEQ_LEN:, 20]  # forward_return_7d
            returns_30d = stock_data[SEQ_LEN:, 21]  # forward_return_30d

            # Filter out NaN/Inf values
            valid_7d = returns_7d[~(torch.isnan(returns_7d) | torch.isinf(returns_7d))]
            valid_30d = returns_30d[~(torch.isnan(returns_30d) | torch.isinf(returns_30d))]

            all_returns_7d.append(valid_7d)
            all_returns_30d.append(valid_30d)

        # Combine all returns for global statistics
        all_returns_7d = torch.cat(all_returns_7d)
        all_returns_30d = torch.cat(all_returns_30d)

        print(f"Collected {len(all_returns_7d)} 7-day returns from {len(csv_files)} stocks")
        print(f"Collected {len(all_returns_30d)} 30-day returns from {len(csv_files)} stocks")
        print(f"7d returns: min={all_returns_7d.min():.4f}, max={all_returns_7d.max():.4f}, mean={all_returns_7d.mean():.4f}, std={all_returns_7d.std():.4f}")
        print(f"30d returns: min={all_returns_30d.min():.4f}, max={all_returns_30d.max():.4f}, mean={all_returns_30d.mean():.4f}, std={all_returns_30d.std():.4f}")

        # STEP 2: Calculate GLOBAL classification thresholds
        print("\n" + "=" * 60)
        print("STEP 2: Calculating GLOBAL classification thresholds...")
        print("=" * 60)

        global_thresholds_7d = calculate_global_thresholds(all_returns_7d)
        global_thresholds_30d = calculate_global_thresholds(all_returns_30d)

    print("Global 7-day return classification thresholds:")
    for name, value in global_thresholds_7d.items():
        print(f"  {name}: {value:.4f}")

    print("Global 30-day return classification thresholds:")
    for name, value in global_thresholds_30d.items():
        print(f"  {name}: {value:.4f}")

    # STEP 3: Create sequences using GLOBAL thresholds (NOT individual percentiles!)
    print("\n" + "=" * 60)
    print("STEP 3: Creating sequences using GLOBAL thresholds...")
    print("=" * 60)

    sequences = []
    total_sequences_checked = 0
    total_sequences_kept = 0

    for filepath in csv_files:
        stock_data = load_stock_data(filepath)
        max_i = len(stock_data) - SEQ_LEN
        file_sequences_checked = 0
        file_sequences_kept = 0

        for i in range(max_i):
            file_sequences_checked += 1
            total_sequences_checked += 1

            # Get sequence and targets
            seq_x = stock_data[i:i+SEQ_LEN, :20]  # All 20 feature columns (0-19)
            forward_return_7d = stock_data[i+SEQ_LEN, 20]  # forward_return_7d column
            forward_return_30d = stock_data[i+SEQ_LEN, 21]  # forward_return_30d column

            # Check for NaN or Inf values in the sequence or targets
            if (torch.isnan(seq_x).any() or torch.isinf(seq_x).any() or
                torch.isnan(forward_return_7d) or torch.isinf(forward_return_7d) or
                torch.isnan(forward_return_30d) or torch.isinf(forward_return_30d)):
                continue  # Skip this sequence

            # *** CORRECTED CLASSIFICATION: Use GLOBAL thresholds ***
            # OLD BROKEN WAY: create_classification_targets(torch.tensor([forward_return_7d]))
            # NEW CORRECT WAY: Use pre-calculated global thresholds
            class_7d = classify_single_return(forward_return_7d, global_thresholds_7d)
            class_30d = classify_single_return(forward_return_30d, global_thresholds_30d)

            sequences.append((seq_x, forward_return_7d, forward_return_30d, class_7d, class_30d))
            file_sequences_kept += 1
            total_sequences_kept += 1

        if file_sequences_checked > 0:
            print(f"{filepath.name}: {file_sequences_kept}/{file_sequences_checked} sequences kept ({(file_sequences_kept/file_sequences_checked)*100:.1f}%)")

    print(f"\nOverall: {total_sequences_kept}/{total_sequences_checked} sequences kept ({(total_sequences_kept/total_sequences_checked)*100:.1f}%)")

    # Print CORRECTED class distribution
    if sequences:
        classes_7d = [seq[3] for seq in sequences]
        classes_30d = [seq[4] for seq in sequences]

        print("\n" + "=" * 60)
        print("CORRECTED 7-day return class distribution:")
        for i, class_name in enumerate(CLASS_NAMES):
            count = classes_7d.count(i)
            print(f"  {class_name}: {count} ({count/len(classes_7d)*100:.1f}%)")

        print("\nCORRECTED 30-day return class distribution:")
        for i, class_name in enumerate(CLASS_NAMES):
            count = classes_30d.count(i)
            print(f"  {class_name}: {count} ({count/len(classes_30d)*100:.1f}%)")
        print("=" * 60)

    # Save to cache for future use
    if use_cache:
        save_global_thresholds_to_cache(global_thresholds_7d, global_thresholds_30d, cache_file)

    return ensure_sequences_on_device(sequences), global_thresholds_7d, global_thresholds_30d


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train multi-task transformer model with CORRECTED global classification')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from (e.g., multitask_transformer_model.pth)')
    parser.add_argument('--epochs', type=int, default=128,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--subset-ratio', type=float, default=0.2,
                       help='Ratio of training data to use per epoch')
    parser.add_argument('--prediction-type', type=str, default='both', choices=['short', 'long', 'both'],
                       help='Type of prediction to train: short-term only, long-term only, or both')
    parser.add_argument('--alpha', type=float, default=1,
                       help='Weight for classification loss (total_loss = regression_loss + alpha * classification_loss)')
    parser.add_argument('--consistency-weight', type=float, default=0.0,
                       help='Weight for consistency loss between regression and classification')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable sequence caching (always regenerate sequences)')
    parser.add_argument('--cache-file', type=str, default='global_thresholds.pkl',
                       help='Path to cache file for storing CORRECTED sequences')

    args = parser.parse_args()

    # Load and prepare data with CORRECTED classification
    print("="*80)
    print("LOADING DATA WITH CORRECTED GLOBAL CLASSIFICATION")
    print("="*80)

    sequences, global_thresholds_7d, global_thresholds_30d = load_all_sequences_corrected(
        use_cache=not args.no_cache,
        cache_file=args.cache_file
    )

    # Shuffle sequences to mix stocks
    random.shuffle(sequences)
    print(f"\nTotal sequences: {len(sequences)}")
    print(f"Sequence shape: {sequences[0][0].shape}")
    print(f"7-day forward return shape: {sequences[0][1].shape}")
    print(f"30-day forward return shape: {sequences[0][2].shape}")
    print(f"7-day classification shape: {sequences[0][3].shape}")
    print(f"30-day classification shape: {sequences[0][4].shape}")
    print(f"Input features: 4 log returns + 10 technical indicators (MA, RSI, MACD, Bollinger Bands, ATR)")
    print(f"Classification uses GLOBAL thresholds (CORRECTED approach)")

    # Split dataset into training and evaluation sets
    train_size = int(0.8 * len(sequences))
    eval_size = len(sequences) - train_size
    train_dataset, eval_dataset = random_split(sequences, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create or load model
    print("\n" + "="*80)
    print("MODEL SETUP")
    print("="*80)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        model = load_trained_model(args.resume)
        print("Loaded existing model weights")
    else:
        print("Starting training from scratch")
        model = create_model()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Define loss functions
    regression_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    # Set loss weights based on prediction type
    short_term_weight = 0.5
    long_term_weight = 0.5
    if args.prediction_type == 'short':
        short_term_weight_regression = 0.7
        long_term_weight_regression = 0.3
        print("Training for 7-DAY forward return prediction only")
    elif args.prediction_type == 'long':
        short_term_weight_regression = 0.3
        long_term_weight_regression = 0.7
        print("Training for 30-DAY forward return prediction only")
    else:  # both
        short_term_weight_regression = 0.5
        long_term_weight_regression = 0.5
        print("Training for BOTH 7-day and 30-day forward return predictions")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop setup
    num_epochs = args.epochs
    subset_ratio = args.subset_ratio
    subset_size = int(subset_ratio * train_size)

    # Track best model and learning rate scheduling
    best_eval_loss = float('inf')
    best_model_path = None

    # Learning rate scheduling parameters
    initial_lr = args.lr
    current_lr = initial_lr
    patience = 3
    patience_counter = 0
    lr_reduction_factor = 0.5
    min_lr = 1e-7
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduction_factor,
                                                    patience=patience, min_lr=min_lr)

    print(f"\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Training for {num_epochs} epochs with learning rate {args.lr}")
    print(f"Using {subset_ratio*100:.0f}% of training data per epoch")
    print(f"7-day return weight: {short_term_weight}, 30-day return weight: {long_term_weight}")
    print(f"Classification loss weight (alpha): {args.alpha}")
    print(f"Consistency loss weight: {args.consistency_weight}")
    print(f"Learning rate scheduling: patience={patience}, reduction_factor={lr_reduction_factor}, min_lr={min_lr}")
    print(f"Cache enabled: {not args.no_cache}")
    print(f"Cache file: {args.cache_file}")
    print(f"Using CORRECTED global classification thresholds!")
    print("="*80)

    for epoch in range(num_epochs):
        model.train()
        subset_indices = random.sample(range(train_size), subset_size)
        train_subset = Subset(train_dataset, subset_indices)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

        total_train_loss = 0
        epoch_regression_7d_loss = 0
        epoch_regression_30d_loss = 0
        epoch_classification_7d_loss = 0
        epoch_classification_30d_loss = 0
        epoch_consistency_loss = 0

        for batch in train_loader:
            x_batch, return_7d_batch, return_30d_batch, class_7d_batch, class_30d_batch = batch

            # Move all batch data to the correct device
            x_batch = x_batch.to(device)
            return_7d_batch = return_7d_batch.to(device)
            return_30d_batch = return_30d_batch.to(device)
            class_7d_batch = class_7d_batch.to(device)
            class_30d_batch = class_30d_batch.to(device)

            # Forward pass
            reg_7d_pred, reg_30d_pred, class_7d_pred, class_30d_pred = model(x_batch)

            # Regression losses
            reg_loss_7d = regression_criterion(reg_7d_pred, return_7d_batch)
            reg_loss_30d = regression_criterion(reg_30d_pred, return_30d_batch)

            # Classification losses
            class_loss_7d = classification_criterion(class_7d_pred, class_7d_batch)
            class_loss_30d = classification_criterion(class_30d_pred, class_30d_batch)

            # Consistency losses
            consistency_7d = consistency_loss(reg_7d_pred, class_7d_pred)
            consistency_30d = consistency_loss(reg_30d_pred, class_30d_pred)

            # Combined losses
            total_reg_loss = (short_term_weight_regression * reg_loss_7d) + (long_term_weight_regression * reg_loss_30d)
            total_class_loss = (short_term_weight * class_loss_7d) + (long_term_weight * class_loss_30d)
            total_consistency_loss = (short_term_weight * consistency_7d) + (long_term_weight * consistency_30d)

            # Final combined loss
            total_loss = total_reg_loss + (args.alpha * total_class_loss) + (args.consistency_weight * total_consistency_loss)

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track losses
            total_train_loss += total_loss.item()
            epoch_regression_7d_loss += reg_loss_7d.item()
            epoch_regression_30d_loss += reg_loss_30d.item()
            epoch_classification_7d_loss += class_loss_7d.item()
            epoch_classification_30d_loss += class_loss_30d.item()
            epoch_consistency_loss += total_consistency_loss.item()

        # Average training losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_reg_7d_loss = epoch_regression_7d_loss / len(train_loader)
        avg_reg_30d_loss = epoch_regression_30d_loss / len(train_loader)
        avg_class_7d_loss = epoch_classification_7d_loss / len(train_loader)
        avg_class_30d_loss = epoch_classification_30d_loss / len(train_loader)
        avg_consistency_loss = epoch_consistency_loss / len(train_loader)

        # Evaluation
        model.eval()
        total_eval_loss = 0
        eval_reg_7d_loss = 0
        eval_reg_30d_loss = 0
        eval_class_7d_loss = 0
        eval_class_30d_loss = 0
        eval_consistency_loss = 0

        # Classification accuracy tracking
        correct_7d = 0
        correct_30d = 0
        total_samples = 0

        confusion_7d = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        confusion_30d = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

        with torch.no_grad():
            for batch in eval_loader:
                x_batch, return_7d_batch, return_30d_batch, class_7d_batch, class_30d_batch = batch

                # Move all batch data to the correct device
                x_batch = x_batch.to(device)
                return_7d_batch = return_7d_batch.to(device)
                return_30d_batch = return_30d_batch.to(device)
                class_7d_batch = class_7d_batch.to(device)
                class_30d_batch = class_30d_batch.to(device)

                reg_7d_pred, reg_30d_pred, class_7d_pred, class_30d_pred = model(x_batch)

                # Regression losses
                reg_loss_7d_eval = regression_criterion(reg_7d_pred, return_7d_batch)
                reg_loss_30d_eval = regression_criterion(reg_30d_pred, return_30d_batch)

                # Classification losses
                class_loss_7d_eval = classification_criterion(class_7d_pred, class_7d_batch)
                class_loss_30d_eval = classification_criterion(class_30d_pred, class_30d_batch)

                # Consistency losses
                consistency_7d_eval = consistency_loss(reg_7d_pred, class_7d_pred)
                consistency_30d_eval = consistency_loss(reg_30d_pred, class_30d_pred)

                # Combined losses
                total_reg_loss_eval = (short_term_weight_regression * reg_loss_7d_eval) + (long_term_weight_regression * reg_loss_30d_eval)
                total_class_loss_eval = (short_term_weight * class_loss_7d_eval) + (long_term_weight * class_loss_30d_eval)
                total_consistency_loss_eval = (short_term_weight * consistency_7d_eval) + (long_term_weight * consistency_30d_eval)

                total_eval_loss += total_reg_loss_eval + (args.alpha * total_class_loss_eval) + (args.consistency_weight * total_consistency_loss_eval)
                eval_reg_7d_loss += reg_loss_7d_eval.item()
                eval_reg_30d_loss += reg_loss_30d_eval.item()
                eval_class_7d_loss += class_loss_7d_eval.item()
                eval_class_30d_loss += class_loss_30d_eval.item()
                eval_consistency_loss += total_consistency_loss_eval.item()

                # Calculate accuracy
                pred_7d = torch.argmax(class_7d_pred, dim=1)
                pred_30d = torch.argmax(class_30d_pred, dim=1)
                correct_7d += (pred_7d == class_7d_batch).sum().item()
                correct_30d += (pred_30d == class_30d_batch).sum().item()
                total_samples += class_7d_batch.size(0)

                # Accumulate confusion matrix for 7d classification
                for t, p in zip(class_7d_batch.view(-1), pred_7d.view(-1)):
                    confusion_7d[t.long(), p.long()] += 1

                for t, p in zip(class_30d_batch.view(-1), pred_30d.view(-1)):
                    confusion_30d[t.long(), p.long()] += 1

        # Average evaluation losses
        avg_eval_loss = total_eval_loss / len(eval_loader)
        avg_eval_reg_7d_loss = eval_reg_7d_loss / len(eval_loader)
        avg_eval_reg_30d_loss = eval_reg_30d_loss / len(eval_loader)
        avg_eval_class_7d_loss = eval_class_7d_loss / len(eval_loader)
        avg_eval_class_30d_loss = eval_class_30d_loss / len(eval_loader)
        avg_eval_consistency_loss = eval_consistency_loss / len(eval_loader)

        # Accuracy
        accuracy_7d = correct_7d / total_samples
        accuracy_30d = correct_30d / total_samples

        # Update learning rate scheduler
        scheduler.step(avg_eval_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save intermediate model for each epoch in a new folder
        intermediate_dir = "intermediate_models_corrected_" + args.prediction_type
        if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)
        intermediate_model_path = os.path.join(intermediate_dir, f"model_epoch_{epoch+1}.pth")
        save_model(model, intermediate_model_path)

        # Check if this is the best model so far
        extra_print = ""
        lr_info = ""
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_model_path = os.path.join(intermediate_dir, "best_model.pth")
            save_model(model, best_model_path)
            extra_print = f" *** NEW BEST ***"
            patience_counter = 0  # Reset patience counter on improvement
        else:
            patience_counter += 1
            if patience_counter >= patience:
                lr_info = f" [LR reduced to {current_lr:.2e}]"

        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_train_loss:.4f} -> {avg_eval_loss:.4f} {extra_print}")
        print(f"  Regression 7d:  {avg_reg_7d_loss:.10f} -> {avg_eval_reg_7d_loss:.10f}")
        print(f"  Regression 30d: {avg_reg_30d_loss:.10f} -> {avg_eval_reg_30d_loss:.10f}")
        print(f"  Classification 7d:  {avg_class_7d_loss:.4f} -> {avg_eval_class_7d_loss:.4f} (Acc: {accuracy_7d:.3f})")
        print(f"  Classification 30d: {avg_class_30d_loss:.4f} -> {avg_eval_class_30d_loss:.4f} (Acc: {accuracy_30d:.3f})")

        # Print confusion matrix summaries for 7d and 30d classification
        print("\tConfusion Matrix Summary (7d):")
        for i, class_name in enumerate(CLASS_NAMES):
            row = confusion_7d[i]
            total = row.sum().item()
            if total > 0:
                percentages = [f"{(count.item()/total)*100:.1f}%" for count in row]
            else:
                percentages = ["0.0%" for _ in row]
            pred_dist = ", ".join([f"{CLASS_NAMES[j]}: {percentages[j]}" for j in range(len(CLASS_NAMES))])
            print(f"\t\t  True {class_name}: {pred_dist}")

        print("\tConfusion Matrix Summary (30d):")
        for i, class_name in enumerate(CLASS_NAMES):
            row = confusion_30d[i]
            total = row.sum().item()
            if total > 0:
                percentages = [f"{(count.item()/total)*100:.1f}%" for count in row]
            else:
                percentages = ["0.0%" for _ in row]
            pred_dist = ", ".join([f"{CLASS_NAMES[j]}: {percentages[j]}" for j in range(len(CLASS_NAMES))])
            print(f"\t\t  True {class_name}: {pred_dist}")

        print(f"  Consistency: {avg_consistency_loss:.4f} -> {avg_eval_consistency_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}, Patience: {patience_counter}/{patience}{lr_info}")

        # Dynamic loss weight adjustment for both prediction types
        if args.prediction_type == 'both':
            # Adjust weights based on relative performance
            reg_performance = (avg_eval_reg_7d_loss + avg_eval_reg_30d_loss) / 2
            class_performance = (avg_eval_class_7d_loss + avg_eval_class_30d_loss) / 2

            # Normalize and adjust alpha
            total_performance = reg_performance + class_performance
            if total_performance > 0:
                new_alpha = class_performance / total_performance
                args.alpha = 0.7 * args.alpha + 0.3 * new_alpha  # Smooth adjustment
                print(f"  Adjusted alpha: {args.alpha:.4f}")

        print("-" * 80)

    # Save the final trained model
    final_model_path = "multitask_transformer_model_corrected_" + args.prediction_type + ".pth"
    save_model(model, final_model_path)

    print(f"\nTraining completed with CORRECTED global classification!")
    print(f"Best evaluation loss: {best_eval_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Final learning rate: {current_lr:.2e}")
    print(f"Final alpha (classification weight): {args.alpha:.4f}")
    print(f"Global thresholds used:")
    print(f"  7d: {global_thresholds_7d}")
    print(f"  30d: {global_thresholds_30d}")

if __name__ == "__main__":
    main()
