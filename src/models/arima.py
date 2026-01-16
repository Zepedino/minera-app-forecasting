"""
Modelos ARIMA y ARIMAX.

Implementa ARIMA univariado con auto_arima.
"""

import pandas as pd
from pmdarima import auto_arima
from config import ARIMA_CONFIG
from src.utils import setup_logger

logger = setup_logger(__name__)


def fit_arima(train_data, forecast_steps, exog_train=None, exog_forecast=None):
    """
    Ajusta modelo ARIMA con seleccion automatica de orden (p,d,q).
    
    Args:
        train_data: Serie temporal de entrenamiento
        forecast_steps: Numero de periodos a pronosticar
        exog_train: Variables exogenas para entrenamiento (opcional, para ARIMAX)
        exog_forecast: Variables exogenas para forecast (opcional, para ARIMAX)
    
    Returns:
        tuple: (forecast, fitted_model)
    """
    config = ARIMA_CONFIG['auto']
    
    # Asegurar frecuencia mensual
    if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
        train_data = train_data.asfreq('MS')
    
    model_type = "ARIMAX" if exog_train is not None else "ARIMA"
    logger.info(f"Entrenando {model_type} con {len(train_data)} observaciones...")
    
    try:
        # Ajustar modelo con auto_arima
        model = auto_arima(
            train_data,
            exogenous=exog_train,
            **config
        )
        
        # Generar forecast
        forecast = model.predict(
            n_periods=forecast_steps,
            exogenous=exog_forecast
        )
        
        # Log orden seleccionado
        order = model.order
        logger.info(f"{model_type} ajustado: orden=(p={order[0]}, d={order[1]}, q={order[2]}), "
                   f"AIC={model.aic():.2f}")
        
        return forecast, model
        
    except Exception as e:
        logger.error(f"Error en {model_type}: {e}")
        raise
