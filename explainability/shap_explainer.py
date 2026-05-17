"""SHAP explainability utilities."""

from __future__ import annotations

from typing import Any


def build_shap_explainer(model: Any, background: Any) -> Any:
    """Create a SHAP explainer for a model and background batch."""

    import shap

    return shap.DeepExplainer(model, background)
