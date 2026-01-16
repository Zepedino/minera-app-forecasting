"""
Router de utilidades.
"""

from .logger import setup_logger
from .data_loader import (
    load_investing_data,
    validate_with_yfinance,
    load_and_validate,
)
from .metrics import (
    calculate_mape,
    calculate_mae,
    calculate_rmse,
    calculate_smape,
    calculate_all_metrics,
    compare_models,
)

__all__ = [
    # Logger
    'setup_logger',
    
    # Data loader
    'load_investing_data',
    'validate_with_yfinance',
    'load_and_validate',
    
    # Metrics
    'calculate_mape',
    'calculate_mae',
    'calculate_rmse',
    'calculate_smape',
    'calculate_all_metrics',
    'compare_models',
]
