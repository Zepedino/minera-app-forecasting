"""
Modelos ARIMAX: ARIMA con variables exógenas
Implementación basada en Balioz et al. (2024) - National Bank of Ukraine
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


class ARIMAXModels:
    """
    ARIMAX: ARIMA con variables exógenas para commodities
    
    Variables exógenas recomendadas (Balioz et al. 2024):
    1. Global Manufacturing PMI (demanda global)
    2. Kilian Real Activity Index (actividad económica real)
    3. USD exchange rate (efecto moneda)
    4. Stock changes (cambios en inventario)
    
    Referencias:
    - Balioz et al. (2024): VAR multivariado para energy commodities
    - Tursoy & Tursoy (2018): Cointegración stock prices vs metal prices
    """
    
    @staticmethod
    def fit_arimax_auto(data: pd.Series,
                        exog: pd.DataFrame,
                        forecast_steps: int,
                        seasonal: bool = True,
                        m: int = 12) -> tuple:
        """
        Auto-ARIMAX: detecta automáticamente parámetros ARIMA + exógenas
        
        Usa pmdarima.auto_arima para búsqueda inteligente de (p,d,q)(P,D,Q)
        
        Parameters
        ----------
        data : pd.Series
            Serie temporal de precios
        exog : pd.DataFrame
            Variables exógenas (PMI, Kilian, USD, etc)
        forecast_steps : int
            Número de periodos a forecasting
        seasonal : bool
            Incluir componente estacional SARIMA
        m : int
            Período de estacionalidad (12 = anual para datos mensuales)
        
        Returns
        -------
        tuple
            (forecast_series, arima_model)
        
        Example
        -------
        >>> # Preparar datos
        >>> exog_df = pd.DataFrame({
        ...     'PMI': pmi_values,
        ...     'Kilian': kilian_values,
        ...     'USD': usd_values
        ... })
        >>> forecast, model = ARIMAXModels.fit_arimax_auto(
        ...     copper_prices,
        ...     exog_df,
        ...     forecast_steps=84,
        ...     seasonal=True,
        ...     m=12
        ... )
        >>> print(f"MAPE: {calculate_mape(test_values, forecast):.2f}%")
        """
        
        # Alineación de datos: exog debe tener al menos misma longitud que data
        if len(exog) < len(data):
            data = data.iloc[-len(exog):]
        elif len(exog) > len(data):
            exog = exog.iloc[-len(data):]
        
        # Auto-ARIMA con variables exógenas
        model = auto_arima(
            data,
            exogenous=exog,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            trace=False,
            max_p=5,
            max_q=5,
            max_P=2,
            max_Q=2,
            max_d=2,
            max_D=1,
            error_action='ignore',
            suppress_warnings=True
        )
        
        # Preparar exógenas para forecast
        # Estrategia: usar último valor de cada variable exógena
        if isinstance(exog, pd.DataFrame):
            forecast_exog = np.tile(
                exog.iloc[-1].values,
                (forecast_steps, 1)
            )
        else:
            forecast_exog = np.tile(exog[-1], (forecast_steps, 1))
        
        # Generar forecast con intervalo de confianza
        forecast_result = model.get_forecast(
            steps=forecast_steps,
            exogenous=forecast_exog
        )
        
        forecast_values = forecast_result.predicted_mean.values
        
        return pd.Series(forecast_values, index=None), model
    
    @staticmethod
    def fit_arimax_manual(data: pd.Series,
                          exog: pd.DataFrame,
                          order: tuple = (1, 1, 1),
                          seasonal_order: tuple = (0, 0, 0, 0),
                          forecast_steps: int = 84) -> tuple:
        """
        ARIMAX manual con orden especificado
        
        Útil cuando se quiere usar orden específico (ej: (1,1,1) baseline)
        o cuando auto_arima no converge
        
        Parameters
        ----------
        data : pd.Series
            Serie temporal
        exog : pd.DataFrame
            Variables exógenas
        order : tuple
            (p, d, q) para ARIMA
        seasonal_order : tuple
            (P, D, Q, s) para SARIMA (default: no estacional)
        forecast_steps : int
            Pasos a forecasting
        
        Returns
        -------
        tuple
            (forecast_series, arima_model)
        
        Example
        -------
        >>> # ARIMAX baseline: (1,1,1) sin estacionalidad
        >>> forecast, model = ARIMAXModels.fit_arimax_manual(
        ...     data,
        ...     exog,
        ...     order=(1,1,1),
        ...     seasonal_order=(0,0,0,0)
        ... )
        """
        
        # Alineación
        if len(exog) < len(data):
            data = data.iloc[-len(exog):]
        elif len(exog) > len(data):
            exog = exog.iloc[-len(data):]
        
        # Fit ARIMAX
        model = ARIMA(
            data,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order
        ).fit()
        
        # Preparar exógenas para forecast
        if isinstance(exog, pd.DataFrame):
            forecast_exog = np.tile(
                exog.iloc[-1].values,
                (forecast_steps, 1)
            )
        else:
            forecast_exog = np.tile(exog[-1], (forecast_steps, 1))
        
        # Forecast
        forecast_result = model.get_forecast(
            steps=forecast_steps,
            exog=forecast_exog
        )
        
        forecast_values = forecast_result.predicted_mean.values
        
        return pd.Series(forecast_values, index=None), model
    
    @staticmethod
    def fit_arimax_with_validation(data: pd.Series,
                                    exog: pd.DataFrame,
                                    test_size: int = 12,
                                    seasonal: bool = True) -> dict:
        """
        ARIMAX con validación cruzada: entrena y valida
        
        Divide datos en train/val, entrena ARIMAX en train,
        valida en val, retorna MAPE de validación
        
        Parameters
        ----------
        data : pd.Series
            Serie temporal completa
        exog : pd.DataFrame
            Variables exógenas
        test_size : int
            Tamaño del conjunto de validación (meses)
        seasonal : bool
            Incluir estacionalidad
        
        Returns
        -------
        dict
            {
                'model': modelo ARIMAX,
                'forecast': predicciones en test,
                'mape': MAPE en conjunto test,
                'parameters': (p,d,q,P,D,Q,s)
            }
        
        Example
        -------
        >>> results = ARIMAXModels.fit_arimax_with_validation(data, exog, test_size=12)
        >>> print(f"Validación MAPE: {results['mape']:.2f}%")
        """
        
        # Split
        data_train = data.iloc[:-test_size]
        data_test = data.iloc[-test_size:]
        exog_train = exog.iloc[:-test_size]
        exog_test = exog.iloc[-test_size:]
        
        # Fit auto_arima en train
        model = auto_arima(
            data_train,
            exogenous=exog_train,
            seasonal=seasonal,
            m=12,
            stepwise=True,
            trace=False,
            suppress_warnings=True
        )
        
        # Forecast en test
        forecast = model.get_forecast(
            steps=test_size,
            exogenous=exog_test
        ).predicted_mean
        
        # Calcular MAPE
        mape = np.mean(np.abs((data_test.values - forecast.values) / data_test.values)) * 100
        
        return {
            'model': model,
            'forecast': pd.Series(forecast.values),
            'mape': mape,
            'parameters': (
                model.order[0], model.order[1], model.order[2],
                model.seasonal_order[0], model.seasonal_order[1],
                model.seasonal_order[2], model.seasonal_order[3]
            )
        }
    
    @staticmethod
    def get_model_summary(model) -> dict:
        """
        Retorna resumen estadístico del modelo ARIMAX
        
        Incluye:
        - AIC/BIC (criterios de información)
        - RMSE
        - Significancia de parámetros
        
        Returns
        -------
        dict
            Resumen de diagnóstico del modelo
        
        Example
        -------
        >>> summary = ARIMAXModels.get_model_summary(model)
        >>> print(f"AIC: {summary['aic']:.2f}, BIC: {summary['bic']:.2f}")
        """
        
        try:
            summary_text = str(model.summary())
            
            return {
                'aic': model.aic,
                'bic': model.bic,
                'rmse': np.sqrt(model.mse),
                'loglik': model.llf,
                'summary': summary_text,
                'parameters': model.params,
                'pvalues': model.pvalues
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def compare_arimax_vs_arima(data: pd.Series,
                                exog: pd.DataFrame,
                                forecast_steps: int) -> dict:
        """
        Compara ARIMAX (con exógenas) vs ARIMA (univariado)
        
        Objetivo: demostrar que ARIMAX mejora ARIMA (Balioz et al. 2024)
        
        Returns
        -------
        dict
            {
                'arimax_mape': MAPE con variables exógenas,
                'arima_mape': MAPE sin variables exógenas,
                'improvement': % de mejora,
                'arimax_model': modelo ARIMAX,
                'arima_model': modelo ARIMA
            }
        
        Example
        -------
        >>> comparison = ARIMAXModels.compare_arimax_vs_arima(data, exog, 84)
        >>> print(f"Mejora ARIMAX: {comparison['improvement_percent']:.1f}%")
        """
        
        # Split train/test
        split_point = int(len(data) * 0.85)
        data_train = data.iloc[:split_point]
        data_test = data.iloc[split_point:]
        exog_train = exog.iloc[:split_point]
        exog_test = exog.iloc[split_point:]
        
        # ARIMAX
        arimax_forecast, arimax_model = ARIMAXModels.fit_arimax_auto(
            data_train, exog_train, len(data_test)
        )
        arimax_mape = np.mean(np.abs(
            (data_test.values - arimax_forecast.values) / data_test.values
        )) * 100
        
        # ARIMA (sin exógenas)
        from pmdarima import auto_arima as auto_arima_univariate
        arima_model = auto_arima_univariate(
            data_train,
            seasonal=True,
            m=12,
            stepwise=True,
            trace=False,
            suppress_warnings=True
        )
        arima_forecast = arima_model.predict(steps=len(data_test))
        arima_mape = np.mean(np.abs(
            (data_test.values - arima_forecast.values) / data_test.values
        )) * 100
        
        improvement = ((arima_mape - arimax_mape) / arima_mape) * 100
        
        return {
            'arimax_mape': arimax_mape,
            'arima_mape': arima_mape,
            'improvement_percent': improvement,
            'arimax_model': arimax_model,
            'arima_model': arima_model
        }
