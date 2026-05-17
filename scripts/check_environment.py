"""Check local environment readiness for the methane pipeline."""

from __future__ import annotations

import importlib.util
import os
import platform


CORE_IMPORTS = ["yaml", "numpy", "pytest"]
FULL_IMPORTS = ["torch", "rasterio", "xarray", "geopandas", "h5py", "zarr", "shap", "timm", "transformers"]


def _status(module_name: str) -> str:
    return "ok" if importlib.util.find_spec(module_name) else "missing"


def main() -> None:
    print(f"Python: {platform.python_version()}")
    if os.environ.get("PIP_NO_INDEX"):
        print("PIP_NO_INDEX: set; pip will not use package indexes unless this is cleared")
    else:
        print("PIP_NO_INDEX: not set")

    print("\nCore imports:")
    for module_name in CORE_IMPORTS:
        print(f"  {module_name}: {_status(module_name)}")

    print("\nFull scientific stack:")
    for module_name in FULL_IMPORTS:
        print(f"  {module_name}: {_status(module_name)}")


if __name__ == "__main__":
    main()
