"""
Modelos de Suavizamiento Exponencial.

Implementa Holt Linear Trend y Damped Trend.
"""

from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
import pandas as pd
from config import EXPONENTIAL_SMOOTHING
from src.utils import setup_logger

logger = setup_logger(__name__)


def fit_holt(train_data, forecast_steps):
    """
    Ajusta modelo Holt Linear Trend.
    
    Args:
        train_data: Serie temporal de entrenamiento
        forecast_steps: Numero de periodos a pronosticar
    
    Returns:
        tuple: (forecast, fitted_model)
    """
    config = EXPONENTIAL_SMOOTHING['holt']
    
    # Asegurar frecuencia mensual
    if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
        train_data = train_data.asfreq('MS')
    
    logger.info(f"Entrenando Holt Linear Trend con {len(train_data)} observaciones...")
    
    try:
        # Separar parametros de init vs fit
        init_params = {k: v for k, v in config.items() if k == 'initialization_method'}
        fit_params = {k: v for k, v in config.items() if k == 'optimized'}
        
        # Ajustar modelo
        model = Holt(train_data, **init_params)
        fitted = model.fit(**fit_params)
        
        # Generar forecast
        forecast = fitted.forecast(steps=forecast_steps)
        
        # Log parametros optimizados
        alpha = fitted.params['smoothing_level']
        beta = fitted.params['smoothing_trend']
        
        logger.info(f"Holt ajustado: alpha={alpha:.4f}, beta={beta:.4f}, "
                   f"AIC={fitted.aic:.2f}")
        
        return forecast, fitted
        
    except Exception as e:
        logger.error(f"Error en Holt: {e}")
        raise


def fit_damped(train_data, forecast_steps):
    """
    Ajusta modelo Damped Trend (Holt con amortiguacion).
    
    Args:
        train_data: Serie temporal de entrenamiento
        forecast_steps: Numero de periodos a pronosticar
    
    Returns:
        tuple: (forecast, fitted_model)
    """
    config = EXPONENTIAL_SMOOTHING['damped']
    
    # Asegurar frecuencia mensual
    if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
        train_data = train_data.asfreq('MS')
    
    logger.info(f"Entrenando Damped Trend con {len(train_data)} observaciones...")
    
    try:
        # Separar parametros de init vs fit
        init_params = {k: v for k, v in config.items() if k != 'optimized'}
        fit_params = {k: v for k, v in config.items() if k == 'optimized'}
        
        # Ajustar modelo
        model = ExponentialSmoothing(train_data, **init_params)
        fitted = model.fit(**fit_params)
        
        # Generar forecast
        forecast = fitted.forecast(steps=forecast_steps)
        
        # Log parametros optimizados
        alpha = fitted.params['smoothing_level']
        beta = fitted.params['smoothing_trend']
        phi = fitted.params.get('damping_trend', None)
        
        phi_str = f"{phi:.4f}" if phi is not None else "N/A"
        
        logger.info(f"Damped ajustado: alpha={alpha:.4f}, beta={beta:.4f}, "
                   f"phi={phi_str}, AIC={fitted.aic:.2f}")
        
        return forecast, fitted
        
    except Exception as e:
        logger.error(f"Error en Damped: {e}")
        raise
