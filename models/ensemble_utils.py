"""
Utilidades para Ensemble de modelos
"""

import pandas as pd
import numpy as np


class EnsembleForecaster:
    """
    Ensemble: combina Prophet + ARIMAX + otros métodos
    """
    
    def __init__(self, models_dict: dict):
        """
        Parameters
        ----------
        models_dict : dict
            {
                'prophet': forecast_series,
                'arimax': forecast_series,
                'exponential_smoothing': forecast_series
            }
        """
        self.models = models_dict
        self.weights = {}
        self.ensemble_forecast = None
    
    def compute_ensemble(self, weights=None, method='weighted_avg'):
        """
        Computa ensemble forecast
        
        Parameters
        ----------
        weights : dict
            Pesos por modelo. Si None, usa pesos iguales
        method : str
            'weighted_avg': promedio ponderado
            'median': mediana robusta
        
        Returns
        -------
        pd.Series
            Forecast del ensemble
        """
        
        if weights is None:
            weights = {k: 1/len(self.models) for k in self.models.keys()}
        
        self.weights = weights
        
        # Stack forecasts
        forecasts_matrix = np.column_stack([
            self.models[name].values for name in self.models.keys()
        ])
        
        if method == 'weighted_avg':
            weight_vector = np.array([
                weights.get(name, 1/len(self.models))
                for name in self.models.keys()
            ])
            self.ensemble_forecast = pd.Series(
                np.average(forecasts_matrix, axis=1, weights=weight_vector)
            )
        
        elif method == 'median':
            self.ensemble_forecast = pd.Series(
                np.median(forecasts_matrix, axis=1)
            )
        
        return self.ensemble_forecast


class MetricsCalculator:
    """
    Calcula métricas de error
    """
    
    @staticmethod
    def calculate_mape(actuals: np.ndarray, forecast: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((actuals - forecast) / (actuals + 1e-8))) * 100
    
    @staticmethod
    def calculate_rmse(actuals: np.ndarray, forecast: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((actuals - forecast) ** 2))
