import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import math

# Model parameters
SEQ_LEN = 252 #200
# Updated INPUT_DIM to include technical indicators
# Columns: Open_log_return, High_log_return, Low_log_return, Close_log_return,
# MA_5d, MA_20d, RSI_14d, MACD_line, MACD_signal, MACD_histogram, BB_upper, BB_middle, BB_lower, ATR_14d
INPUT_DIM = 0  # 4 log returns + 16 technical indicators
MODEL_DIM = 256
NUM_LAYERS = 2
NUM_HEADS = 4
BATCH_SIZE = 32
DROPOUT = 0.2


def set_input_dim(input_dim):
    global INPUT_DIM
    INPUT_DIM = input_dim

def set_model_feature_and_target_columns(filepath):
    df = pd.read_csv(filepath)
    target_column_names = ["forward_return_7d", "forward_return_30d"]
    filter_column_names = ["Date"]
    feature_columns = [col for col in df.columns if col not in target_column_names and col not in filter_column_names]
    set_input_dim(len(feature_columns))
    return feature_columns, target_column_names

# Classification parameters
NUM_CLASSES = 5  # big_down, down, no_change, up, big_up
CLASS_NAMES = ['big_down', 'down', 'no_change', 'up', 'big_up']

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    """Get the current device (GPU/CPU)"""
    return device

def print_device_info():
    """Print current device information"""
    print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    """
    Enhanced positional encoding that combines learned and sinusoidal encodings.
    Supports variable sequence lengths and provides better inductive bias.
    """
    def __init__(self, model_dim, max_len=5000, dropout=0.1, use_sinusoidal=True, use_learned=True):
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.max_len = max_len
        self.use_sinusoidal = use_sinusoidal
        self.use_learned = use_learned

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Initialize encoding components
        self.sinusoidal_encoding = None
        self.learned_encoding = None

        if use_sinusoidal:
            # Create sinusoidal positional encoding
            pe = torch.zeros(max_len, model_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, model_dim)

            # Register as buffer (not parameter) so it's not updated during training
            self.register_buffer('sinusoidal_encoding', pe)

        if use_learned:
            # Learned positional encoding
            self.learned_encoding = nn.Parameter(torch.randn(max_len, 1, model_dim))

        # Scaling factor for combining encodings
        if use_sinusoidal and use_learned:
            self.combine_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Learnable weights

    def forward(self, x, seq_len=None):
        """
        Apply positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, model_dim)
            seq_len: Optional sequence length (if different from x.size(1))

        Returns:
            Tensor with positional encoding added
        """
        if seq_len is None:
            seq_len = x.size(1)

        # Ensure sequence length doesn't exceed max_len
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")

        # Initialize positional encoding
        pos_encoding = torch.zeros_like(x)

        if self.use_sinusoidal and self.sinusoidal_encoding is not None:
            # Add sinusoidal encoding
            sinusoidal_pe = self.sinusoidal_encoding[:seq_len, :, :].transpose(0, 1)  # (1, seq_len, model_dim)
            pos_encoding += sinusoidal_pe

        if self.use_learned and self.learned_encoding is not None:
            # Add learned encoding
            learned_pe = self.learned_encoding[:seq_len, :, :].transpose(0, 1)  # (1, seq_len, model_dim)
            pos_encoding += learned_pe

        # Apply dropout for regularization
        pos_encoding = self.dropout(pos_encoding)

        return pos_encoding

def create_classification_targets(returns, percentiles=(20, 80)):
    """
    Create classification targets based on return percentiles.

    Args:
        returns: Tensor of return values
        percentiles: Tuple of (low, high) percentiles for class boundaries

    Returns:
        Tensor of class labels (0: big_down, 1: down, 2: no_change, 3: up, 4: big_up)
    """
    # Ensure returns are on the correct device
    returns = returns.to(device)

    # Calculate percentile thresholds
    low_thresh = torch.quantile(returns, percentiles[0] / 100.0)
    high_thresh = torch.quantile(returns, percentiles[1] / 100.0)

    # Create class labels on the same device
    classes = torch.zeros_like(returns, dtype=torch.long, device=device)

    # Define thresholds for 5 classes
    very_low = torch.quantile(returns, 0.1)  # Bottom 10%
    very_high = torch.quantile(returns, 0.9)  # Top 10%

    classes[returns <= very_low] = 0      # big_down
    classes[(returns > very_low) & (returns <= low_thresh)] = 1  # down
    classes[(returns > low_thresh) & (returns < high_thresh)] = 2  # no_change
    classes[(returns >= high_thresh) & (returns < very_high)] = 3  # up
    classes[returns >= very_high] = 4     # big_up

    return classes

class AttentionPooling(nn.Module):
    def __init__(self, model_dim, dropout=0.2):
        super(AttentionPooling, self).__init__()
        self.query_vector = nn.Parameter(torch.randn(model_dim))
        self.scale = model_dim ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, model_dim)
        weights = torch.matmul(x, self.query_vector) / self.scale
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        attended_output = torch.sum(x * weights, dim=1)
        attended_output = self.dropout(attended_output)
        return attended_output

# Enhanced Transformer Model with Multi-task Learning
class MultiTaskTransformerModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, model_dim=MODEL_DIM, num_layers=NUM_LAYERS,
                 nhead=NUM_HEADS, dropout=DROPOUT, num_classes=NUM_CLASSES,
                 seq_len=SEQ_LEN, use_sinusoidal_pe=True, use_learned_pe=True):
        super(MultiTaskTransformerModel, self).__init__()

        # Enhanced positional encoding
        self.pos_encoder = PositionalEncoding(
            model_dim=model_dim,
            max_len=seq_len * 2,  # Allow for longer sequences
            dropout=dropout * 0.5,  # Lower dropout for positional encoding
            use_sinusoidal=use_sinusoidal_pe,
            use_learned=use_learned_pe
        )

        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead,
                                                   batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Separate attention pooling for different tasks
        self.attention_regression_7d = AttentionPooling(model_dim, dropout)
        self.attention_regression_30d = AttentionPooling(model_dim, dropout)
        self.attention_classification_7d = AttentionPooling(model_dim, dropout)
        self.attention_classification_30d = AttentionPooling(model_dim, dropout)

        # Regression heads
        self.fc_regression_7d = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1)
        )
        self.fc_regression_30d = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1)
        )

        # Classification heads
        self.fc_classification_7d = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )
        self.fc_classification_30d = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

    def forward(self, src):
        # Shared encoder with enhanced positional encoding
        src = self.input_proj(src)
        pos_encoding = self.pos_encoder(src)
        src = src + pos_encoding  # Add positional encoding

        encoder_output = self.transformer_encoder(src)

        # Task-specific attention pooling
        regression_context_7d = self.attention_regression_7d(encoder_output)
        regression_context_30d = self.attention_regression_30d(encoder_output)
        classification_context_7d = self.attention_classification_7d(encoder_output)
        classification_context_30d = self.attention_classification_30d(encoder_output)

        # Predictions
        regression_7d = self.fc_regression_7d(regression_context_7d).squeeze(-1)
        regression_30d = self.fc_regression_30d(regression_context_30d).squeeze(-1)
        classification_7d = self.fc_classification_7d(classification_context_7d)
        classification_30d = self.fc_classification_30d(classification_context_30d)

        return regression_7d, regression_30d, classification_7d, classification_30d

def consistency_loss(regression_pred, classification_pred, temperature=1.0):
    """
    Consistency loss to ensure regression and classification predictions align.

    Args:
        regression_pred: Regression predictions
        classification_pred: Classification logits
        temperature: Temperature for softmax

    Returns:
        Consistency loss value
    """
    # Convert regression to directional signal
    regression_direction = torch.sign(regression_pred)

    # Convert classification to expected direction
    class_probs = torch.softmax(classification_pred / temperature, dim=-1)
    # Expected direction: weighted sum of class directions
    class_directions = torch.tensor([-2, -1, 0, 1, 2], device=classification_pred.device, dtype=torch.float32)
    expected_direction = torch.sum(class_probs * class_directions, dim=-1)
    expected_direction = torch.sign(expected_direction)

    # Consistency loss: penalize when directions don't match
    consistency = torch.mean((regression_direction - expected_direction) ** 2)
    return consistency

def load_stock_data(filepath,
                    data_dir='../stock_technical_data',
                    feature_columns=[],
                    target_columns=[]):
    """
    Load stock return data with technical indicators from CSV file.

    Args:
        filepath: Path to CSV file or stock symbol
        data_dir: Directory containing stock return data files with technical indicators

    Returns:
        torch.Tensor: Stock data with shape (num_days, 16)
                     Columns: [Open_log_return, High_log_return, Low_log_return, Close_log_return,
                              MA_5d, MA_20d, RSI_14d, MACD_line, MACD_signal, MACD_histogram,
                              BB_upper, BB_middle, BB_lower, ATR_14d, forward_return_7d, forward_return_30d]
    """
    if isinstance(filepath, str) and not filepath.endswith('.csv'):
        # If it's a symbol, construct the file path
        filepath = Path(data_dir) / f'{filepath}.csv'

    df = pd.read_csv(filepath)
    print(f"Loaded stock data with technical indicators: {filepath}")
    df.sort_values('Date', inplace=True)

    # Combine features and targets
    all_columns = feature_columns + target_columns
    data = df[all_columns].values
    return torch.tensor(data, dtype=torch.float32).to(device)

def create_model(seq_len=SEQ_LEN, use_sinusoidal_pe=True, use_learned_pe=True):
    """
    Create and initialize a new multi-task transformer model.

    Args:
        seq_len: Sequence length for positional encoding
        use_sinusoidal_pe: Whether to use sinusoidal positional encoding
        use_learned_pe: Whether to use learned positional encoding

    Returns:
        MultiTaskTransformerModel: Initialized model on the correct device
    """
    model = MultiTaskTransformerModel(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        nhead=NUM_HEADS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        seq_len=seq_len,
        use_sinusoidal_pe=use_sinusoidal_pe,
        use_learned_pe=use_learned_pe
    )
    model = model.to(device)

    # Proper weight initialization
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return model

def load_trained_model(model_path, seq_len=SEQ_LEN, use_sinusoidal_pe=True, use_learned_pe=True):
    """
    Load a trained model from file.

    Args:
        model_path (str): Path to the saved model file
        seq_len: Sequence length for positional encoding
        use_sinusoidal_pe: Whether to use sinusoidal positional encoding
        use_learned_pe: Whether to use learned positional encoding

    Returns:
        MultiTaskTransformerModel: Loaded model
    """
    model = MultiTaskTransformerModel(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        nhead=NUM_HEADS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        seq_len=seq_len,
        use_sinusoidal_pe=use_sinusoidal_pe,
        use_learned_pe=use_learned_pe
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def save_model(model, filepath):
    """
    Save a model to file.

    Args:
        model: The model to save
        filepath (str): Path where to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to: {filepath}")

def interpret_predictions(regression_pred, classification_pred, return_type="7d"):
    """
    Interpret model predictions for both regression and classification.

    Args:
        regression_pred: Regression predictions
        classification_pred: Classification logits
        return_type: "7d" or "30d"

    Returns:
        Dict with interpreted predictions
    """
    # Classification probabilities
    class_probs = torch.softmax(classification_pred, dim=-1)
    predicted_classes = torch.argmax(class_probs, dim=-1)

    # Convert to class names
    predicted_class_names = [CLASS_NAMES[cls.item()] for cls in predicted_classes]

    # Confidence scores
    confidence_scores = torch.max(class_probs, dim=-1)[0]

    return {
        'regression_prediction': regression_pred.cpu().numpy(),
        'classification_prediction': predicted_class_names,
        'classification_probabilities': class_probs.cpu().numpy(),
        'confidence_scores': confidence_scores.cpu().numpy(),
        'return_type': return_type
    }


def classify_single_return(return_value, global_thresholds):
    """
    Classify a single return value using pre-calculated global thresholds.

    Args:
        return_value: Single return value (float or tensor)
        global_thresholds: Dict with keys 'very_low', 'low', 'high', 'very_high'

    Returns:
        Integer class label (0: big_down, 1: down, 2: no_change, 3: up, 4: big_up)
    """
    if isinstance(return_value, torch.Tensor):
        return_value = return_value.item()

    very_low = global_thresholds['very_low']
    low = global_thresholds['low']
    high = global_thresholds['high']
    very_high = global_thresholds['very_high']

    if return_value <= very_low:
        return 0  # big_down
    elif return_value <= low:
        return 1  # down
    elif return_value < high:
        return 2  # no_change
    elif return_value < very_high:
        return 3  # up
    else:
        return 4  # big_up

def calculate_global_thresholds(all_returns, percentiles=(40, 60)):
    """
    Calculate global classification thresholds from all returns.

    Args:
        all_returns: Tensor containing all return values from all stocks
        percentiles: Tuple of (low, high) percentiles for class boundaries

    Returns:
        Dictionary with threshold values
    """
    all_returns = all_returns.to(device)

    thresholds = {
        'very_low': torch.quantile(all_returns, 0.2).item(),  # Bottom 20%
        'low': torch.quantile(all_returns, percentiles[0] / 100.0).item(),  # Bottom 40%
        'high': torch.quantile(all_returns, percentiles[1] / 100.0).item(),  # Top 60%
        'very_high': torch.quantile(all_returns, 0.8).item(),  # Top 20%
    }

    return thresholds
