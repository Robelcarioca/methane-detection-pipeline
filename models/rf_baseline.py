"""Random Forest Baseline Model."""

from __future__ import annotations
import numpy as np
import h5py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(hdf5_path: str):
    """Loads and flattens HDF5 patch data for the Random Forest."""
    with h5py.File(hdf5_path, 'r') as f:
        X = np.array(f['x'])  # Expected shape: (N, 20, 128, 128)
        y = np.array(f['y'])  # Expected shape: (N, 1, 128, 128)
        
        N, C, H, W = X.shape
        # Flatten spatial dimensions for RF: (N * H * W, Channels)
        X_flat = X.transpose(0, 2, 3, 1).reshape(-1, C)
        y_flat = y.reshape(-1)
        
        return X_flat, y_flat

def train_evaluate_rf(X_train, X_test, y_train, y_test):
    """Trains and evaluates the Random Forest baseline."""
    print("Training Random Forest (this might take a moment)...")
    
    # n_jobs=-1 tells the model to use all your computer's CPU cores to train faster
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = rf.predict(X_test)

    # Calculate validation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("-" * 30)
    print("Baseline Model Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print("-" * 30)
    
    return rf

if __name__ == "__main__":
    # --- DUMMY DATA MODE ---
    # When real data is ready, delete this block and uncomment the real data loader below.
    print("Using dummy data (20 features) matching config.yaml...")
    X_dummy = np.random.rand(10000, 20)  # 10,000 pixels, 20 satellite bands
    y_dummy = np.random.rand(10000)      # 10,000 target methane values

    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)
    # -----------------------
    
    """
    # --- REAL DATA MODE (Uncomment when ready) ---
    print("Loading HDF5 data...")
    X, y = load_data("../datasets/niger_delta_train.h5")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """

    # Run the pipeline
    trained_model = train_evaluate_rf(X_train, X_test, y_train, y_test)