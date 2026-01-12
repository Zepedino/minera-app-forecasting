"""
Modelos de Suavizamiento Exponencial
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ETSModel

class ExponentialSmoothingModels:
    """
    Implementa los 3 modelos de suavizamiento exponencial:
    1. Holt (tendencia lineal)
    2. Brown (doble exponencial)
    3. Damped Trend (tendencia amortiguada)
    """
    
    @staticmethod
    def fit_holt(train, forecast_steps):
        """
        Holt Linear Trend (Tendencia Lineal)
        
        Args:
            train: Serie de entrenamiento
            forecast_steps: Pasos a pronosticar
            
        Returns:
            forecast: Pronósticos
            fitted_model: Modelo ajustado
        """
        try:
            model = Holt(train, initialization_method="estimated")
            fitted_model = model.fit(optimized=True)
            
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            return forecast, fitted_model
            
        except Exception as e:
            print(f"❌ Error en Holt: {str(e)}")
            return None, None
    
    @staticmethod
    def fit_brown(train, forecast_steps):
        """
        Brown Double Exponential Smoothing
        (Equivalente a Holt sin tendencia optimizada)
        
        Args:
            train: Serie de entrenamiento
            forecast_steps: Pasos a pronosticar
            
        Returns:
            forecast, fitted_model
        """
        try:
            # Brown es SimpleExpSmoothing aplicado 2 veces
            # O Holt con parámetros específicos
            model = ExponentialSmoothing(
                train,
                trend='add',
                seasonal=None,
                initialization_method="estimated"
            )
            
            fitted_model = model.fit(
                optimized=True
            )

            # Para ver qué parámetros eligió:
            alpha = fitted_model.params['smoothing_level']
            beta = fitted_model.params['smoothing_trend']
            print(f"   📊 Parámetros optimizados: α={alpha:.3f}, β={beta:.3f}")
            
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            return forecast, fitted_model
            
        except Exception as e:
            print(f"❌ Error en Brown: {str(e)}")
            return None, None
    
    @staticmethod
    def fit_damped_trend(train, forecast_steps):
        """
        Damped Trend (Tendencia Amortiguada)
        RECOMENDADO para horizontes largos (5 años)
        
        Args:
            train: Serie de entrenamiento
            forecast_steps: Pasos a pronosticar
            
        Returns:
            forecast, fitted_model
        """
        try:
            model = ExponentialSmoothing(
                train,
                trend='add',
                damped_trend=True,  # ⭐ Característica clave
                seasonal=None,
                initialization_method="estimated"
            )
            
            fitted_model = model.fit(optimized=True)
            
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            return forecast, fitted_model
            
        except Exception as e:
            print(f"❌ Error en Damped Trend: {str(e)}")
            return None, None
    
    @staticmethod
    def get_model_params(fitted_model):
        """
        Extrae parámetros del modelo ajustado
        
        Returns:
            dict con parámetros
        """
        try:
            params = {
                'alpha': fitted_model.params.get('smoothing_level', None),
                'beta': fitted_model.params.get('smoothing_trend', None),
                'phi': fitted_model.params.get('damping_trend', None)
            }
            return params
        except:
            return {}
