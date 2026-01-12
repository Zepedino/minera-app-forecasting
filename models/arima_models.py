"""
Modelos ARIMA y SARIMA
"""
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class ARIMAModels:
    """
    Implementa modelos ARIMA:
    1. ARIMA automático (univariado)
    2. SARIMA automático (con estacionalidad)
    3. ARIMA manual (para control fino)
    """
    
    @staticmethod
    def fit_auto_arima(train, forecast_steps, seasonal=False, m=12):
        """
        ARIMA Automático usando pmdarima
        
        Args:
            train: Serie de entrenamiento
            forecast_steps: Pasos a pronosticar
            seasonal: Si usar SARIMA
            m: Periodicidad estacional (12 para mensual)
            
        Returns:
            forecast, fitted_model
        """
        try:
            print(f"   🔍 Buscando mejor modelo ARIMA{'X' if seasonal else ''}...")
            
            # Convertir a numpy array para evitar problemas con índices
            train_values = train.values if hasattr(train, 'values') else np.array(train)
            
            # Remover NaN si existen
            train_values = train_values[~np.isnan(train_values)]
            
            model = auto_arima(
                train_values,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                d=None,  # Auto-detectar diferenciación
                seasonal=seasonal,
                m=m if seasonal else 1,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                D=None,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                n_jobs=-1,
                information_criterion='aic',
                maxiter=50
            )
            
            print(f"   ✅ Mejor modelo: {model.order}" + 
                  (f" x {model.seasonal_order}" if seasonal else ""))
            
            # Generar pronóstico
            forecast = model.predict(n_periods=forecast_steps)
            
            # Convertir a Series para consistencia
            forecast_series = pd.Series(forecast, index=range(len(forecast)))
            
            return forecast_series, model
            
        except Exception as e:
            print(f"   ❌ Error en Auto-ARIMA: {str(e)}")
            return None, None
    
    @staticmethod
    def fit_manual_arima(train, forecast_steps, order=(1,1,1)):
        """
        ARIMA Manual con parámetros específicos
        
        Args:
            train: Serie de entrenamiento
            forecast_steps: Pasos a pronosticar
            order: Tupla (p, d, q)
            
        Returns:
            forecast, fitted_model
        """
        try:
            # Convertir a numpy array
            train_values = train.values if hasattr(train, 'values') else np.array(train)
            
            # Remover NaN si existen
            train_values = train_values[~np.isnan(train_values)]
            
            # Crear y ajustar modelo
            model = ARIMA(train_values, order=order)
            fitted_model = model.fit()
            
            # Generar pronóstico
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            # Convertir a Series
            forecast_series = pd.Series(forecast, index=range(len(forecast)))
            
            return forecast_series, fitted_model
            
        except Exception as e:
            print(f"   ❌ Error en ARIMA manual {order}: {str(e)}")
            return None, None
    
    @staticmethod
    def get_model_summary(fitted_model):
        """
        Obtiene resumen del modelo
        
        Returns:
            dict con información del modelo
        """
        try:
            summary = {
                'order': fitted_model.order if hasattr(fitted_model, 'order') else 'N/A',
                'aic': fitted_model.aic() if hasattr(fitted_model, 'aic') else 'N/A',
                'bic': fitted_model.bic() if hasattr(fitted_model, 'bic') else 'N/A'
            }
            
            if hasattr(fitted_model, 'seasonal_order'):
                summary['seasonal_order'] = fitted_model.seasonal_order
            
            return summary
        except:
            return {}
    
    @staticmethod
    def diagnose_model(fitted_model, train_data=None):
        """
        Diagnóstico del modelo ARIMA
        
        Args:
            fitted_model: Modelo ajustado
            train_data: Datos de entrenamiento (opcional)
            
        Returns:
            dict con métricas de diagnóstico
        """
        try:
            diagnostics = {}
            
            # AIC y BIC
            if hasattr(fitted_model, 'aic'):
                diagnostics['aic'] = fitted_model.aic()
            if hasattr(fitted_model, 'bic'):
                diagnostics['bic'] = fitted_model.bic()
            
            # Orden del modelo
            if hasattr(fitted_model, 'order'):
                diagnostics['order'] = fitted_model.order
            
            # Coeficientes
            if hasattr(fitted_model, 'params'):
                diagnostics['n_params'] = len(fitted_model.params())
            
            # Residuos
            if hasattr(fitted_model, 'resid'):
                residuals = fitted_model.resid()
                diagnostics['residual_mean'] = np.mean(residuals)
                diagnostics['residual_std'] = np.std(residuals)
            
            return diagnostics
            
        except Exception as e:
            print(f"   ⚠️  No se pudo generar diagnóstico: {str(e)}")
            return {}
    
    @staticmethod
    def forecast_with_confidence_interval(fitted_model, forecast_steps, alpha=0.05):
        """
        Genera pronóstico con intervalos de confianza
        
        Args:
            fitted_model: Modelo ajustado
            forecast_steps: Pasos a pronosticar
            alpha: Nivel de significancia (0.05 = 95% confianza)
            
        Returns:
            dict con forecast, lower_bound, upper_bound
        """
        try:
            # Pronóstico
            forecast = fitted_model.predict(n_periods=forecast_steps)
            
            # Intervalos de confianza (aproximación simple)
            # En producción, usar fitted_model.predict(return_conf_int=True)
            std_error = np.std(fitted_model.resid()) if hasattr(fitted_model, 'resid') else np.std(forecast) * 0.15
            z_score = 1.96  # Para 95% confianza
            
            lower_bound = forecast - z_score * std_error
            upper_bound = forecast + z_score * std_error
            
            return {
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence_level': 1 - alpha
            }
            
        except Exception as e:
            print(f"   ⚠️  Error calculando intervalos de confianza: {str(e)}")
            return {
                'forecast': fitted_model.predict(n_periods=forecast_steps) if hasattr(fitted_model, 'predict') else None,
                'lower_bound': None,
                'upper_bound': None,
                'confidence_level': None
            }
