"""
Metricas de evaluacion de modelos de forecasting.

Calcula MAPE, MAE, RMSE, SMAPE para comparar performance.
"""

import numpy as np
import pandas as pd
from .logger import setup_logger

logger = setup_logger(__name__)


def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        MAPE en porcentaje
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Evitar division por cero
    mask = y_true != 0
    
    if not mask.any():
        logger.warning("Todos los valores reales son cero, MAPE no definido")
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        MAE
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        RMSE
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def calculate_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.
    
    Menos sensible a valores extremos que MAPE.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        SMAPE en porcentaje
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred))
    
    # Evitar division por cero
    mask = denominator != 0
    
    if not mask.any():
        logger.warning("Denominador cero en SMAPE")
        return np.nan
    
    smape = np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
    return smape


def calculate_all_metrics(y_true, y_pred, model_name=""):
    """
    Calcula todas las metricas y retorna diccionario.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        model_name: Nombre del modelo (para logging)
    
    Returns:
        Diccionario con todas las metricas
    """
    metrics = {
        'MAPE': calculate_mape(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'SMAPE': calculate_smape(y_true, y_pred),
    }
    
    if model_name:
        logger.info(f"Metricas {model_name}: MAPE={metrics['MAPE']:.2f}%, "
                   f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
    
    return metrics


def compare_models(results_dict):
    """
    Compara multiples modelos y retorna tabla ordenada por MAPE.
    
    Args:
        results_dict: Diccionario {model_name: {'y_true': [...], 'y_pred': [...]}}
    
    Returns:
        DataFrame con metricas de todos los modelos, ordenado por MAPE
    """
    comparison = []
    
    for model_name, data in results_dict.items():
        metrics = calculate_all_metrics(data['y_true'], data['y_pred'], model_name)
        metrics['Model'] = model_name
        comparison.append(metrics)
    
    df = pd.DataFrame(comparison)
    df = df[['Model', 'MAPE', 'MAE', 'RMSE', 'SMAPE']]  # Reordenar columnas
    df = df.sort_values('MAPE')  # Ordenar por mejor MAPE
    
    logger.info(f"\n=== COMPARACION DE MODELOS ===\n{df.to_string(index=False)}")
    
    return df
