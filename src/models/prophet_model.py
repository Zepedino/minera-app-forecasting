"""
Modelo Prophet para forecasting de commodities.

Implementa Prophet de Facebook con configuracion del config/model_config.py.
Referencia: Taylor & Letham (2018) - Forecasting at Scale.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from config import PROPHET_CONFIG
from src.utils import setup_logger

logger = setup_logger(__name__)


def fit_prophet(train_data, forecast_steps, regressors_train=None, regressors_forecast=None):
    """
    Ajusta modelo Prophet para series de tiempo de commodities.
    
    Args:
        train_data: Serie temporal de entrenamiento (pd.Series con DatetimeIndex)
        forecast_steps: Numero de periodos a pronosticar
        regressors_train: DataFrame con variables exogenas para entrenamiento (opcional)
        regressors_forecast: DataFrame con variables exogenas para forecast (opcional)
    
    Returns:
        tuple: (forecast, fitted_model)
        
    Notas:
        - Usa configuracion de PROPHET_CONFIG['base']
        - Datos deben ser mensuales
        - Si hay regresores, ambos dataframes deben proporcionarse
    """
    config = PROPHET_CONFIG['base']
    
    # Preparar datos en formato Prophet (ds, y)
    df_train = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data.values
    })
    
    model_type = "Prophet-X" if regressors_train is not None else "Prophet"
    logger.info(f"Entrenando {model_type} con {len(train_data)} observaciones...")
    
    try:
        # Inicializar modelo
        model = Prophet(
            yearly_seasonality=config['yearly_seasonality'],
            weekly_seasonality=config['weekly_seasonality'],
            daily_seasonality=config['daily_seasonality'],
            seasonality_mode=config['seasonality_mode'],
            n_changepoints=config['n_changepoints'],
            changepoint_prior_scale=config['changepoint_prior_scale'],
            interval_width=config['interval_width'],
            uncertainty_samples=config['uncertainty_samples']
        )
        
        # Agregar regresores si existen
        if regressors_train is not None:
            if regressors_forecast is None:
                raise ValueError("Si proporcionas regressors_train, debes proporcionar regressors_forecast")
            
            for col in regressors_train.columns:
                model.add_regressor(col)
                logger.info(f"  Regresor agregado: {col}")
            
            # Combinar con regresores de entrenamiento
            df_train = df_train.merge(
                regressors_train, 
                left_on='ds', 
                right_index=True, 
                how='left'
            )
        
        # Ajustar modelo
        model.fit(df_train)
        
        # Crear dataframe futuro para forecast
        future = model.make_future_dataframe(periods=forecast_steps, freq='MS')  # MS = Month Start
        
        # Si hay regresores, agregarlos al futuro
        if regressors_train is not None:
            # Combinar datos historicos + forecast de regresores
            all_regressors = pd.concat([regressors_train, regressors_forecast])
            
            for col in regressors_train.columns:
                future[col] = future['ds'].map(all_regressors[col])
        
        # Generar forecast
        forecast_df = model.predict(future)
        
        # Extraer solo las predicciones futuras (forecast_steps)
        forecast = forecast_df['yhat'].iloc[-forecast_steps:].values
        
        # Log informacion
        logger.info(f"{model_type} ajustado: changepoints={config['n_changepoints']}, "
                   f"prior_scale={config['changepoint_prior_scale']}, "
                   f"seasonality={config['seasonality_mode']}")
        
        return forecast, model
        
    except Exception as e:
        logger.error(f"Error en {model_type}: {e}")
        raise


def fit_prophet_conservative(train_data, forecast_steps):
    """
    Prophet con configuracion conservadora (baja flexibilidad).
    
    Util para commodities muy volatiles donde queremos evitar sobreajuste.
    
    Args:
        train_data: Serie temporal de entrenamiento
        forecast_steps: Numero de periodos a pronosticar
    
    Returns:
        tuple: (forecast, fitted_model)
    """
    # Crear configuracion conservadora
    config_conservative = PROPHET_CONFIG['base'].copy()
    config_conservative['changepoint_prior_scale'] = 0.01  # Muy rigido
    config_conservative['n_changepoints'] = 5  # Pocos changepoints
    
    # Modificar temporalmente config global
    original_config = PROPHET_CONFIG['base'].copy()
    PROPHET_CONFIG['base'] = config_conservative
    
    try:
        result = fit_prophet(train_data, forecast_steps)
    finally:
        # Restaurar config original
        PROPHET_CONFIG['base'] = original_config
    
    return result


def fit_prophet_flexible(train_data, forecast_steps):
    """
    Prophet con configuracion flexible (alta adaptabilidad).
    
    Util para commodities estables con tendencias claras.
    
    Args:
        train_data: Serie temporal de entrenamiento
        forecast_steps: Numero de periodos a pronosticar
    
    Returns:
        tuple: (forecast, fitted_model)
    """
    # Crear configuracion flexible
    config_flexible = PROPHET_CONFIG['base'].copy()
    config_flexible['changepoint_prior_scale'] = 0.10  # Mas flexible
    config_flexible['n_changepoints'] = 20  # Mas changepoints
    
    # Modificar temporalmente config global
    original_config = PROPHET_CONFIG['base'].copy()
    PROPHET_CONFIG['base'] = config_flexible
    
    try:
        result = fit_prophet(train_data, forecast_steps)
    finally:
        # Restaurar config original
        PROPHET_CONFIG['base'] = original_config
    
    return result
