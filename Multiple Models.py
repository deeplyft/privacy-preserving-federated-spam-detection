"""Compatibility entry point for the updated model-comparison workflow.

Run `python compare_models.py` for the current implementation. This file is
kept so the original project filename still works without obsolete pandas and
scikit-learn calls.
"""
from compare_models import main


if __name__ == "__main__":
    main()
