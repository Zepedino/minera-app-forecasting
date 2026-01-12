"""
Métricas para evaluar modelos de pronóstico
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    
    MAPE < 10%: Excelente
    MAPE 10-20%: Bueno
    MAPE > 20%: Aceptable
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Evitar división por cero
    mask = y_true != 0
    
    if not mask.any():
        return np.inf
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def calculate_mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    Más robusto que MAPE cuando hay valores cercanos a cero
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Evitar división por cero
    mask = denominator != 0
    
    if not mask.any():
        return np.inf
    
    smape = np.mean(numerator[mask] / denominator[mask]) * 100
    return smape

def evaluate_forecast(y_true, y_pred, model_name="Modelo"):
    """
    Calcula todas las métricas de evaluación
    
    Returns:
        dict con métricas
    """
    metrics = {
        'Modelo': model_name,
        'MAPE': calculate_mape(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'SMAPE': calculate_smape(y_true, y_pred)
    }
    
    return metrics

def print_metrics(metrics):
    """Imprime métricas de forma legible"""
    print(f"\n📈 Métricas de {metrics['Modelo']}:")
    print(f"   MAPE:  {metrics['MAPE']:.2f}%")
    print(f"   MAE:   {metrics['MAE']:.2f}")
    print(f"   RMSE:  {metrics['RMSE']:.2f}")
    print(f"   SMAPE: {metrics['SMAPE']:.2f}%")
    
    # Interpretación del MAPE
    if metrics['MAPE'] < 10:
        calidad = "🟢 Excelente"
    elif metrics['MAPE'] < 20:
        calidad = "🟡 Bueno"
    else:
        calidad = "🟠 Aceptable"
    
    print(f"   Calidad: {calidad}")
