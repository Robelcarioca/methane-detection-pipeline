"""SHAP explainability utilities."""

from __future__ import annotations
from typing import Any, List
import os
import matplotlib.pyplot as plt
import shap
import torch

def build_shap_explainer(model: Any, background: Any) -> Any:
    """Create a SHAP explainer for a model and background batch."""
    return shap.DeepExplainer(model, background)

def generate_summary_plot(
    explainer: Any, 
    test_data: torch.Tensor, 
    feature_names: List[str], 
    save_dir: str = "../visualization"
) -> str:
    """
    Generates a SHAP summary plot to show feature importance 
    across satellite bands and saves it as a high-res PNG.
    """
    # Calculate SHAP values for the test batch
    shap_values = explainer.shap_values(test_data)
    
    # Unwrap the SHAP values if PyTorch returns them as a list
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    # Squeeze out any extra dimensions (e.g., turning (10, 20, 1) into (10, 20))
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) > 2:
        shap_values = shap_values.squeeze()
    # ---------------
    
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate the SHAP summary plot
    plt.figure(figsize=(10, 6))
    
    shap.summary_plot(
        shap_values, 
        test_data.cpu().numpy(), 
        feature_names=feature_names, 
        show=False
    )
    
    # Save the plot at 300 DPI for scientific publication
    output_path = os.path.join(save_dir, "shap_feature_importance.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path