"""
Router de modelos de forecasting.
"""

from .exponential_smoothing import fit_holt, fit_damped
from .arima import fit_arima

__all__ = [
    'fit_holt',
    'fit_damped',
    'fit_arima',
]
