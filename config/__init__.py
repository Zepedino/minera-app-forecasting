"""
Router de configuracion.

Permite importar todo desde config directamente:
    from config import EXPONENTIAL_SMOOTHING, PROJECT_ROOT
"""

# Importar configuraciones de modelos
from .model_config import (
    EXPONENTIAL_SMOOTHING,
    ARIMA_CONFIG,
    ARIMAX_CONFIG,
    PROPHET_CONFIG,
    MONTE_CARLO_CONFIG,
    PROJECT_INFO,
    TRAIN_TEST_SPLIT,
)

# Importar rutas y tickers
from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXOGENOUS_DATA_DIR,
    RESULTS_DIR,
    BASELINE_RESULTS_DIR,
    MONTECARLO_RESULTS_DIR,
    ADVANCED_RESULTS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    DATA_FILES,
    YFINANCE_TICKERS,
    YFINANCE_EXOGENOUS,
    FRED_SERIES,
    EXOGENOUS_BY_METAL,
)

# Definir exports
__all__ = [
    # Configuraciones de modelos
    'EXPONENTIAL_SMOOTHING',
    'ARIMA_CONFIG',
    'ARIMAX_CONFIG',
    'PROPHET_CONFIG',
    'MONTE_CARLO_CONFIG',
    'PROJECT_INFO',
    'TRAIN_TEST_SPLIT',
    
    # Rutas
    'PROJECT_ROOT',
    'DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'EXOGENOUS_DATA_DIR',
    'RESULTS_DIR',
    'BASELINE_RESULTS_DIR',
    'MONTECARLO_RESULTS_DIR',
    'ADVANCED_RESULTS_DIR',
    'REPORTS_DIR',
    'LOGS_DIR',
    
    # Mapeos
    'DATA_FILES',
    'YFINANCE_TICKERS',
    'YFINANCE_EXOGENOUS',
    'FRED_SERIES',
    'EXOGENOUS_BY_METAL',
]
