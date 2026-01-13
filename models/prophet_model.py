"""
Modelos Prophet para detección de cambios estructurales en commodities
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ProphetModels:
    """
    Implementación de Facebook Prophet con automatic changepoint detection
    
    Referencias:
    - Taylor & Letham (2018): Prophet paper
    - Zherlitsyn et al. (2025): Prophet + ML hybrid para commodities
    """
    
    @staticmethod
    def fit_prophet_base(data: pd.Series, forecast_steps: int) -> tuple:
        """
        Prophet base con automatic changepoint detection
        
        Detecta automáticamente:
        - COVID-19 (2020): caída 30-50%
        - Ukraine war (2022): rally 50-100%
        - China lockdown (2022): caída 10-20%
        
        Parameters
        ----------
        data : pd.Series
            Serie temporal de precios (index=fechas, values=precios)
        forecast_steps : int
            Número de periodos a forecasting
        
        Returns
        -------
        tuple
            (forecast_series, prophet_model)
        
        Example
        -------
        >>> copper_prices = pd.Series(data_values, index=date_index)
        >>> forecast, model = ProphetModels.fit_prophet_base(copper_prices, 84)
        >>> print(f"MAPE: {calculate_mape(test_values, forecast):.2f}%")
        """
        
        # Preparar datos en formato Prophet (ds, y)
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Crear modelo con detección automática de changepoints
        # changepoint_prior_scale controla sensibilidad de detección
        model = Prophet(
            changepoint_prior_scale=0.05,      # Detectar cambios significativos (default: 0.05)
            interval_width=0.90,                # IC 90% (P5-P95)
            yearly_seasonality=True,            # Capturar ciclos anuales
            seasonality_mode='additive',        # Modo aditivo (mejor para commodities)
            seasonality_prior_scale=10,         # Peso de seasonalidad
            daily_seasonality=False,            # Sin seasonalidad diaria (data mensual)
            weekly_seasonality=False            # Sin seasonalidad semanal
        )
        
        # Fit del modelo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        
        # Generar forecast
        future = model.make_future_dataframe(periods=forecast_steps, freq='MS')
        forecast = model.predict(future)
        
        # Extraer predicciones (últimos forecast_steps)
        forecast_values = forecast['yhat'].iloc[-forecast_steps:].values
        
        return pd.Series(forecast_values, index=None), model
    
    @staticmethod
    def fit_prophet_with_holidays(data: pd.Series, 
                                   forecast_steps: int,
                                   holidays_dict: pd.DataFrame = None) -> tuple:
        """
        Prophet con holidays/shocks (eventos extremos)
        
        Captura:
        - COVID-19 crash (2020-03): -30% a -50%
        - Ukraine war (2022-02): rally +50% a +100%
        - China lockdown (2022-09): caída -10% a -20%
        - Restricciones ambientales
        - Cambios en regulación minera
        
        Parameters
        ----------
        data : pd.Series
            Serie temporal
        forecast_steps : int
            Pasos a forecasting
        holidays_dict : pd.DataFrame
            DataFrame con columnas ['holiday', 'ds', 'lower_window', 'upper_window']
            lower_window: días antes del evento
            upper_window: días después del evento
        
        Returns
        -------
        tuple
            (forecast_series, prophet_model)
        
        Example
        -------
        >>> # Definir eventos históricos
        >>> holidays = pd.DataFrame({
        ...     'holiday': ['COVID_crash', 'Ukraine_war'],
        ...     'ds': pd.to_datetime(['2020-03-15', '2022-02-24']),
        ...     'lower_window': [-30, -10],
        ...     'upper_window': [90, 120]
        ... })
        >>> forecast, model = ProphetModels.fit_prophet_with_holidays(data, 84, holidays)
        """
        
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Holidays por defecto si no se especifican
        if holidays_dict is None:
            holidays_dict = pd.DataFrame({
                'holiday': ['COVID_crash', 'Ukraine_war', 'China_lockdown'],
                'ds': pd.to_datetime(['2020-03-15', '2022-02-24', '2022-09-01']),
                'lower_window': [-30, -10, -15],
                'upper_window': [90, 120, 60]
            })
        
        model = Prophet(
            changepoint_prior_scale=0.05,
            interval_width=0.90,
            yearly_seasonality=True,
            seasonality_mode='additive',
            holidays=holidays_dict
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        
        future = model.make_future_dataframe(periods=forecast_steps, freq='MS')
        forecast = model.predict(future)
        
        forecast_values = forecast['yhat'].iloc[-forecast_steps:].values
        
        return pd.Series(forecast_values, index=None), model
    
    @staticmethod
    def get_changepoint_dates(model: Prophet) -> pd.DatetimeIndex:
        """
        Retorna fechas de cambios estructurales detectados automáticamente
        
        Útil para análisis post-hoc de qué eventos Prophet detectó
        
        Returns
        -------
        pd.DatetimeIndex
            Fechas de changepoints detectados
        
        Example
        -------
        >>> forecast, model = ProphetModels.fit_prophet_base(data, 84)
        >>> changepoints = ProphetModels.get_changepoint_dates(model)
        >>> print(f"Cambios detectados: {changepoints}")
        """
        return model.changepoints
    
    @staticmethod
    def get_forecast_intervals(model: Prophet, 
                              forecast_steps: int) -> tuple:
        """
        Retorna intervalo de confianza completo (P5, P50, P95)
        
        Returns
        -------
        tuple
            (forecast_mean, lower_bound, upper_bound)
        
        Example
        -------
        >>> mean, lower, upper = ProphetModels.get_forecast_intervals(model, 84)
        >>> print(f"P95: {upper[-1]:.2f} | P50: {mean[-1]:.2f} | P5: {lower[-1]:.2f}")
        """
        
        future = model.make_future_dataframe(periods=forecast_steps, freq='MS')
        forecast = model.predict(future)
        
        forecast_mean = forecast['yhat'].iloc[-forecast_steps:].values
        lower = forecast['yhat_lower'].iloc[-forecast_steps:].values
        upper = forecast['yhat_upper'].iloc[-forecast_steps:].values
        
        return forecast_mean, lower, upper
