"""
Carga y validacion de datos de precios.

Carga CSVs de Investing.com y valida contra Yahoo Finance.
"""

import pandas as pd
import yfinance as yf
from config import DATA_FILES, YFINANCE_TICKERS, DATA_DIR
from .logger import setup_logger

logger = setup_logger(__name__)


def load_investing_data(metal: str) -> pd.Series:
    """
    Carga datos de Investing.com para un metal especifico.
    
    Args:
        metal: Nombre del metal ('cobre', 'oro', 'plata', 'cobalto')
    
    Returns:
        Serie temporal con precios mensuales
    """
    if metal not in DATA_FILES:
        raise ValueError(f"Metal '{metal}' no valido. Opciones: {list(DATA_FILES.keys())}")
    
    file_path = DATA_FILES[metal]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    # Leer CSV
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Extraer columna de precio
    price_series = df['Price']
    
    logger.info(f"Cargado {metal}: {len(price_series)} observaciones "
                f"({price_series.index.min():%Y-%m-%d} a {price_series.index.max():%Y-%m-%d})")
    
    return price_series


def validate_with_yfinance(metal: str, investing_data: pd.Series) -> dict:
    """
    Valida datos de Investing.com contra Yahoo Finance.
    
    NOTA: Yahoo Finance solo tiene datos desde ~2000 para commodities futures.
    Los datos de Investing.com desde 1990 NO se validan (se usan igual).
    
    Args:
        metal: Nombre del metal
        investing_data: Serie temporal de Investing.com
    
    Returns:
        Diccionario con metricas de validacion
    """
    if metal not in YFINANCE_TICKERS:
        logger.warning(f"{metal} no tiene ticker en Yahoo Finance")
        return {'validated': False, 'reason': 'No ticker disponible'}
    
    ticker = YFINANCE_TICKERS[metal]
    
    # Yahoo Finance: datos desde ~2000 en adelante
    start = pd.Timestamp('2000-08-01')
    end = investing_data.index.max()
    
    logger.info(f"Validando con Yahoo Finance ({ticker}) desde {start:%Y-%m-%d}...")
    
    try:
        # Descargar datos de Yahoo Finance
        yf_data = yf.download(ticker, start=start, end=end, progress=False)
        
        if yf_data.empty:
            logger.warning(f"Yahoo Finance no devolvio datos para {ticker}")
            return {'validated': False, 'reason': 'Sin datos de Yahoo Finance'}
        
        # Asegurar que es Serie (no DataFrame multi-columna)
        if isinstance(yf_data, pd.DataFrame):
            if 'Close' in yf_data.columns:
                yf_data = yf_data['Close']
            else:
                yf_data = yf_data.iloc[:, 0]  # Primera columna
        
        # Eliminar NaN
        yf_data = yf_data.dropna()
        
        # Resample a mensual (inicio de mes)
        yf_monthly = yf_data.resample('MS').last().dropna()
        
        # Filtrar investing_data al rango de Yahoo Finance
        investing_filtered = investing_data[investing_data.index >= start].copy()
        
        # Normalizar indices a inicio de mes
        investing_filtered.index = pd.to_datetime(investing_filtered.index).to_period('M').to_timestamp()
        investing_filtered = investing_filtered.dropna()
        
        # Fechas comunes
        common_dates = investing_filtered.index.intersection(yf_monthly.index)
        
        if len(common_dates) < 10:
            logger.warning(f"Solo {len(common_dates)} fechas comunes - validacion poco confiable")
            return {'validated': False, 'reason': f'Solo {len(common_dates)} fechas comunes'}
        
        # Extraer valores comunes (aplanar a 1D)
        inv_common = investing_filtered.loc[common_dates].values.flatten()
        yf_common = yf_monthly.loc[common_dates].values.flatten()
        
        # Calcular correlacion
        corr = pd.Series(inv_common).corr(pd.Series(yf_common))
        
        # Calcular diferencia media porcentual
        mean_diff_pct = (abs(inv_common - yf_common) / yf_common * 100).mean()
        
        logger.info(f"Validacion {metal}: correlacion={corr:.4f}, "
                   f"diff_media={mean_diff_pct:.2f}%, "
                   f"fechas_comunes={len(common_dates)}/{len(investing_filtered)}")
        
        return {
            'validated': True,
            'correlation': corr,
            'mean_diff_pct': mean_diff_pct,
            'common_dates': len(common_dates),
            'total_dates': len(investing_data),
        }
        
    except Exception as e:
        logger.error(f"Error en validacion Yahoo Finance: {e}")
        return {'validated': False, 'reason': str(e)}


def load_and_validate(metal_name: str) -> pd.Series:
    logger.info(f"=== Cargando datos de {metal_name.upper()} ===")
    
    if metal_name not in DATA_FILES:
        raise ValueError(f"Metal '{metal_name}' no configurado. Opciones: {list(DATA_FILES.keys())}")
    
    filepath = DATA_FILES[metal_name]
    
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    if df['Price'].dtype == 'object':
        df['Price'] = df['Price'].str.replace(',', '')
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    df = df.dropna(subset=['Price'])
    
    df = df.set_index('Date').sort_index()
    
    df = df[~df.index.duplicated(keep='last')]
    
    series = df['Price']
    
    series = series[series.index <= '2025-12-31']
    
    series = series.asfreq('MS')
    
    if series.isna().any():
        nan_count = series.isna().sum()
        logger.warning(f"{nan_count} NaN detectados, rellenando...")
        series = series.ffill()
        if series.isna().any():
            series = series.bfill()
    
    logger.info(f"Cargado {metal_name}: {len(series)} observaciones ({series.index.min().date()} a {series.index.max().date()})")
    
    if metal_name in YFINANCE_TICKERS:
        validate_with_yfinance(metal_name, series)
    else:
        logger.warning(f"{metal_name} no tiene ticker en Yahoo Finance")
    
    return series
