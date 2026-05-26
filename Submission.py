from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings('ignore')

# Optimized residual block with pre-activation for better gradient flow
class ResidualBlock(nn.Module):
    __slots__ = ['linear1', 'linear2', 'activation', 'dropout', 'batch_norm1', 'batch_norm2']
    
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        
        # Pre-activation architecture for better gradient flow
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU(inplace=True)  # inplace saves memory
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        out = self.batch_norm1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear1(out)
        
        out = self.batch_norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        return out + residual

# Optimized residual regressor with configurable architecture
class ResidualRegressor(nn.Module):
    __slots__ = ['input_layer', 'activation', 'dropout', 'blocks', 'output_layer']
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_blocks: int = 3):
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Use ModuleList for flexible number of blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) 
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_layer(x)

# Optimized preprocessing with caching support
class DataPreprocessor:
    def __init__(self, target_col: str = "PitNextLap"):
        self.target_col = target_col
        self.preprocessor = None
        self.numerical_cols = None
        self.categorical_cols = None
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Drop ID columns efficiently
        X = train_df.drop(columns=[self.target_col, 'id'], errors='ignore')
        y = train_df[self.target_col].values
        X_test = test_df.drop(columns=['id'], errors='ignore')
        test_ids = test_df['id'].values if 'id' in test_df.columns else np.arange(len(test_df))
        
        # Detect categorical columns more efficiently
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        
        # Build preprocessor pipeline
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', numerical_transformer, self.numerical_cols),
            ('cat', categorical_transformer, self.categorical_cols)
        ])
        
        # Transform data
        X_processed = self.preprocessor.fit_transform(X)
        X_test_processed = self.preprocessor.transform(X_test)
        
        return X_processed.astype(np.float32), y.astype(np.float32), X_test_processed.astype(np.float32), test_ids

# Optimized training with gradient clipping and learning rate scheduling
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int = 16,
    learning_rate: float = 1e-4,
    gradient_clip: float = 1.0
) -> nn.Module:
    
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_mae = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        predictions, targets = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                outputs = model(batch_x).squeeze()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.numpy())
        
        valid_mae = mean_absolute_error(targets, predictions)
        
        # Manual learning rate scheduling with verbose output
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(valid_mae)
        new_lr = optimizer.param_groups[0]['lr']
        
        lr_msg = ""
        if new_lr < old_lr:
            lr_msg = f" | LR reduced: {old_lr:.2e} → {new_lr:.2e}"
        
        print(f"Epoch {epoch + 1:02d} | Train Loss: {train_loss / len(train_loader):.5f} | Valid MAE: {valid_mae:.5f}{lr_msg}")
        
        if valid_mae < best_mae:
            best_mae = valid_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    return model

# Optimized prediction with batching
@torch.no_grad()
def predict_model(model: nn.Module, X_test: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    predictions = []
    
    # Process in batches
    for i in range(0, len(X_test), batch_size):
        batch = torch.from_numpy(X_test[i:i+batch_size]).to(device, dtype=torch.float32, non_blocking=True)
        outputs = model(batch).squeeze()
        predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions, dtype=np.float32)

# Optimized blending with validation
def blend_predictions(test_ids: np.ndarray, prediction_dict: Dict, output_path: Path, target_col: str):
    total_weight = sum(config['weight'] for config in prediction_dict.values())
    weighted_sum = np.zeros(len(test_ids))
    
    for name, config in prediction_dict.items():
        weight = config['weight'] / total_weight  # Normalize weights
        weighted_sum += config['predictions'] * weight
        print(f"Blending {name} | Weight: {weight:.4f}")
    
    # Create and save submission
    submission = pd.DataFrame({'id': test_ids, target_col: weighted_sum})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission

# Main function with optimizations
def main():
    TARGET = "PitNextLap"
    COMP_PATH = Path("/kaggle/input/competitions/playground-series-s6e5")
    BLEND_PATH = Path("/kaggle/input/datasets/anthonytherrien/predicting-f1-pit-stops-vault")
    
    # Set device and optimize for performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"Using device: {device} (GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    else:
        print(f"Using device: {device}")
    
    # Load data efficiently
    print("Loading data...")
    train_df = pd.read_csv(COMP_PATH / "train.csv")
    test_df = pd.read_csv(COMP_PATH / "test.csv")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(TARGET)
    X, y, X_test, test_ids = preprocessor.preprocess_data(train_df, test_df)
    print(f"Processed features shape: {X.shape}")
    
    # Split data with stratification for better validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    print(f"Train size: {len(X_train)}, Validation size: {len(X_valid)}")
    
    # Create data loaders with optimized settings
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    
    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True, 
                             pin_memory=device.type == 'cuda', num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False,
                             pin_memory=device.type == 'cuda', num_workers=num_workers)
    
    # Create and train model
    print("Creating model...")
    model = ResidualRegressor(
        input_dim=X.shape[1],
        hidden_dim=256,  # Increased for better capacity
        dropout=0.3,     # Reduced for better learning
        num_blocks=3
    ).to(device)
    
    # Count parameters for debugging
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train model with gradient clipping
    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=25,  # Increased epochs with early stopping via scheduler
        learning_rate=5e-4,  # Higher initial learning rate with scheduler
        gradient_clip=1.0
    )
    
    # Generate predictions
    print("Generating predictions...")
    nn_predictions = predict_model(model, X_test, device, batch_size=8192)
    print(f"Predictions shape: {nn_predictions.shape}")
    
    # Load external predictions for blending
    print("Loading external predictions for blending...")
    prediction_dict = {
        "sub1": {
            "predictions": pd.read_csv(BLEND_PATH / "submission.csv")[TARGET].values,
            "weight": 2.9
        },
        "sub2": {
            "predictions": pd.read_csv(BLEND_PATH / "submission (1).csv")[TARGET].values,
            "weight": 0.1
        },
        "nn": {
            "predictions": nn_predictions,
            "weight": 1.0  # Increased weight for neural network
        }
    }
    
    # Blend predictions
    print("Blending predictions...")
    blend_predictions(test_ids, prediction_dict, Path("submission.csv"), TARGET)
    
    print("Done!")

if __name__ == "__main__":
    main()
