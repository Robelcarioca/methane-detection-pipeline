"""
=============================================================================
EXECUTION INSTRUCTIONS 
=============================================================================
This script contains the training loop and validation integration. 

To run this with the actual pipeline:
1. Update the `MethaneDataset` class below to point to the actual 
   local or server directories where the final processed tensors are stored.
2. Adjust the dummy hyperparameters (Epochs, LR, Batch Size) in 
   `train_model` to your actual optimized values.
3. Ensure you save the model weights at the end of the script 
   so the team can run the final test evaluation on the withheld dataset.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import your custom adapter and metrics
from models.transfer_learning import MethaneTransferModel
from utils.metrics import evaluate_batch

class MethaneDataset(Dataset):
    """
    Custom Dataset to load Robel's [20, 128, 128] tensors.
    """
    def __init__(self, data_paths, label_paths):
        # [TODO: ROBEL] DUMMY DATA VARS: `data_paths` and `label_paths`.
        # Replace these with the actual directory paths to the final processed datasets.
        self.data_paths = data_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # [TODO: ROBEL] DATA LOADING LOGIC: 
        # Update this based on your final file format. 
        # If saved as .pt, use torch.load(). If .npy, use np.load() and convert to tensor.
        x = torch.load(self.data_paths[idx])  # Expected actual data: [20, 128, 128] float tensor
        y = torch.load(self.label_paths[idx]) # Expected actual data: [1, 128, 128] binary tensor
        return x, y

def train_model(train_loader, val_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Training on device: {device}")
    
    # 1. Initialize your custom model (Stage 1: Backbone Frozen by default)
    # Verify this matches Robel's final tensor depth before running.
    model = MethaneTransferModel(in_channels=20, out_channels=1, freeze_backbone=True).to(device)
    
    # We use BCEWithLogitsLoss because your model outputs raw logits (no sigmoid at the end).
    # It is mathematically much more stable than doing Sigmoid -> BCELoss.
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer specifically for the un-frozen layers in Stage 1
    # DUMMY HYPERPARAMETER: `lr=1e-3`. Replace with your tuned LR.
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # DUMMY HYPERPARAMETERS: Epoch counts.
    # Replace these 3 and 7 with your actual planned epochs for warm-up and fine-tuning.
    EPOCHS_STAGE_1 = 3
    EPOCHS_STAGE_2 = 7
    
    # --- STAGE 1: WARM-UP ---
    print("\n--- Starting Stage 1: Frozen Backbone (Warm-up) ---")
    for epoch in range(EPOCHS_STAGE_1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS_STAGE_1} | Loss: {train_loss/len(train_loader):.4f}")
        
    # --- TRANSITION TO STAGE 2 ---
    print("\n--- Starting Stage 2: Unfreezing Backbone (Fine-Tuning) ---")
    for param in model.parameters():
        param.requires_grad = True  # Unfreeze everything!
        
    # Create a new optimizer for all layers with a much lower learning rate
    # DUMMY HYPERPARAMETER: `lr=1e-5`. Replace with your tuned fine-tuning LR.
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    # --- STAGE 2: FINE-TUNING ---
    for epoch in range(EPOCHS_STAGE_2):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDATION STEP (Using Spatial Metrics) ---
        model.eval()
        with torch.no_grad():
            # Grab one batch of validation data to test
            val_x, val_y = next(iter(val_loader))
            val_x, val_y = val_x.to(device), val_y.to(device)
            
            val_outputs = model(val_x)
            
            # Convert logits to probabilities via Sigmoid before sending to your evaluator
            probabilities = torch.sigmoid(val_outputs).cpu().numpy()
            ground_truth = val_y.cpu().numpy()
            
            # Plug straight into the evaluator you built yesterday!
            val_metrics = evaluate_batch(ground_truth, probabilities, threshold=0.5)
            
        print(f"Epoch {epoch+1}/{EPOCHS_STAGE_2} | Loss: {train_loss/len(train_loader):.4f} | Val IoU: {val_metrics['macro_active_iou']:.4f}")

    # EXPORT REQUIRED:
    # Add your torch.save() logic here so the final model weights are exported.
    # Need this file to run the final Evaluation on the withheld Test Dataset.
    # e.g., torch.save(model.state_dict(), 'methane_model_final.pth')

    return model