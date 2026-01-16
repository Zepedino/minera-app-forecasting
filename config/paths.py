"""
Rutas centralizadas del proyecto.
"""

from pathlib import Path

# Raiz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent

# Subdirectorios principales
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"

# Datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXOGENOUS_DATA_DIR = DATA_DIR / "exogenous"  # Para variables exogenas

# Resultados
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
MONTECARLO_RESULTS_DIR = RESULTS_DIR / "montecarlo"
ADVANCED_RESULTS_DIR = RESULTS_DIR / "advanced"
REPORTS_DIR = RESULTS_DIR / "reports"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Mapeo de archivos de datos (CSV raw)
DATA_FILES = {
    'cobre': RAW_DATA_DIR / 'Copper-Futures-Historical-Data.csv',
    'oro': RAW_DATA_DIR / 'Gold-Futures-Historical-Data.csv',
    'plata': RAW_DATA_DIR / 'Silver-Futures-Historical-Data.csv',
    'cobalto': RAW_DATA_DIR / 'Cobalt-Futures-Historical-Data.csv',
}

# Tickers de Yahoo Finance (para validacion de precios)
YFINANCE_TICKERS = {
    'cobre': 'HG=F',
    'oro': 'GC=F',
    'plata': 'SI=F',
}

# Variables exogenas disponibles en Yahoo Finance
YFINANCE_EXOGENOUS = {
    'usd_index': 'DX-Y.NYB',
    'vix': '^VIX',
    'brent_oil': 'BZ=F',
    'wti_oil': 'CL=F',
    'nickel': 'NI=F',
    'gold': 'GC=F',  # Para usar como exogena de plata
}

# Variables exogenas de FRED (requiere API key)
FRED_SERIES = {
    'china_pmi': 'CHNMFGPMISMEI',  # China Manufacturing PMI
    'real_rates': 'DFII10',  # 10-Year Real Interest Rate
}

# Mapeo de exogenas por metal
EXOGENOUS_BY_METAL = {
    'cobre': ['usd_index', 'china_pmi', 'brent_oil'],
    'oro': ['usd_index', 'real_rates', 'vix', 'brent_oil'],
    'plata': ['usd_index', 'gold', 'vix', 'brent_oil'],
    'cobalto': ['nickel', 'usd_index', 'brent_oil'],  # ev_sales manual
}

def ensure_directories_exist():
    """Crea directorios necesarios."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXOGENOUS_DATA_DIR,
        BASELINE_RESULTS_DIR,
        MONTECARLO_RESULTS_DIR,
        ADVANCED_RESULTS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Ejecutar al importar
ensure_directories_exist()
