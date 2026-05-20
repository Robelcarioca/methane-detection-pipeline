import os
import torch
import torch.nn as nn
from explainability.shap_explainer import build_shap_explainer, generate_summary_plot

# 1. Update the DummyModel to accept 20 channels
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 1) # Changed from 5 to 20

    def forward(self, x):
        return self.linear(x)

model = DummyModel()
model.eval()

# 2. Update dummy data to have 20 features
background_data = torch.randn(100, 20)
test_data = torch.randn(10, 20)

# 3. Update feature names based on config.yaml (Total of 20 to match the channels)
feature_names = [
    "B11 (SWIR 1)", "B12 (SWIR 2)", "B04 (Red)", "B03 (Green)", "B02 (Blue)",
    "SWIR Ratio (B12/B11)", 
    "Temporal Diff (B11)", "Temporal Diff (B12)", "Temporal Diff (B04)", "Temporal Diff (B03)", 
    "Texture (B11)", "Texture (B12)", "Texture (B04)",
    "Cloud Mask", "Water Mask", "Valid Data Mask",
    "Aux Band 1", "Aux Band 2", "Aux Band 3", "Aux Band 4" # Placeholders for remaining channels
]

print("Building SHAP explainer...")
explainer = build_shap_explainer(model, background_data)

print("Generating summary plot...")
output_path = generate_summary_plot(
    explainer=explainer,
    test_data=test_data,
    feature_names=feature_names,
    save_dir="visualization"
)

print(f"Success! SHAP plot saved to: {os.path.abspath(output_path)}")