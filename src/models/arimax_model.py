"""
Modelo ARIMAX para predicción de precios de metales
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

logger = setup_logger(__name__)


class ARIMAXForecaster:
    """
    Modelo ARIMAX con selección automática de parámetros
    """
    
    def __init__(self, metal_name, order=(1,1,1), seasonal_order=(0,0,0,0)):
        """
        Args:
            metal_name: Nombre del metal
            order: (p, d, q) para ARIMA
            seasonal_order: (P, D, Q, s) para componente estacional
        """
        self.metal_name = metal_name
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
        
    def check_stationarity(self, series):
        """Test de estacionariedad (ADF)"""
        result = adfuller(series.dropna())
        logger.info(f"  ADF Statistic: {result[0]:.4f}")
        logger.info(f"  p-value: {result[1]:.4f}")
        
        if result[1] < 0.05:
            logger.info(f"  Serie estacionaria (p < 0.05)")
            return True
        else:
            logger.warning(f"  Serie NO estacionaria (p >= 0.05)")
            return False
    
    def fit(self, df_train, exog_cols=None):
        """
        Entrena modelo ARIMAX
        
        Args:
            df_train: DataFrame con columna 'Price' y variables exógenas
            exog_cols: Lista de columnas exógenas a usar (None = todas excepto Price)
        """
        logger.info(f"\n=== Entrenando ARIMAX: {self.metal_name.upper()} ===")
        
        # Preparar datos
        y_train = df_train['Price']
        
        if exog_cols is None:
            exog_cols = [col for col in df_train.columns if col != 'Price']
        
        if len(exog_cols) == 0:
            logger.warning("  Sin variables exógenas, usando ARIMA puro")
            X_train = None
        else:
            X_train = df_train[exog_cols]
            logger.info(f"  Variables exógenas: {exog_cols}")
        
        logger.info(f"  Observaciones: {len(y_train)}")
        logger.info(f"  Orden ARIMA: {self.order}")
        
        # Test de estacionariedad
        logger.info("\n  Test de Estacionariedad:")
        self.check_stationarity(y_train)
        
        # Entrenar modelo
        try:
            self.model = SARIMAX(
                y_train,
                exog=X_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.results = self.model.fit(disp=False, maxiter=200)
            
            logger.info(f"\n  Modelo entrenado exitosamente")
            logger.info(f"  AIC: {self.results.aic:.2f}")
            logger.info(f"  BIC: {self.results.bic:.2f}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"  Error entrenando modelo: {e}")
            return None
    
    def predict(self, df_test, exog_cols=None):
        """
        Genera predicciones
        
        Args:
            df_test: DataFrame con variables exógenas para forecast
            exog_cols: Lista de columnas exógenas (debe coincidir con fit)
        
        Returns:
            Series con predicciones
        """
        if self.results is None:
            logger.error("Modelo no entrenado. Ejecuta fit() primero")
            return None
        
        if exog_cols is None:
            exog_cols = [col for col in df_test.columns if col != 'Price']
        
        if len(exog_cols) == 0:
            X_test = None
        else:
            X_test = df_test[exog_cols]
        
        # Forecast
        try:
            forecast = self.results.get_forecast(
                steps=len(df_test),
                exog=X_test
            )
            
            predictions = forecast.predicted_mean
            predictions.index = df_test.index
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generando predicciones: {e}")
            return None
    
    def evaluate(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        metrics = {
            'MAPE': mape,
            'MAE': mae,
            'RMSE': rmse
        }
        
        logger.info(f"\n  === METRICAS ===")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  MAE:  ${mae:.2f}")
        logger.info(f"  RMSE: ${rmse:.2f}")
        
        return metrics


def run_arimax_experiment(metal_name, test_size=24):
    """
    Ejecuta experimento completo ARIMAX para un metal
    
    Args:
        metal_name: 'cobre', 'oro', 'plata', 'cobalto'
        test_size: Meses para test (default 24 = 2 años)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENTO ARIMAX: {metal_name.upper()}")
    logger.info(f"{'='*60}")
    
    # 1. Cargar datos combinados
    data_path = Path(f'data/processed/{metal_name}_with_exogenous.csv')
    
    if not data_path.exists():
        logger.error(f"Datos no encontrados. Ejecuta merge_exogenous.py primero")
        return None
    
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # 2. Split train/test
    split_idx = len(df) - test_size
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    logger.info(f"\nTrain: {df_train.index[0]} a {df_train.index[-1]} ({len(df_train)} obs)")
    logger.info(f"Test:  {df_test.index[0]} a {df_test.index[-1]} ({len(df_test)} obs)")
    
    # 3. Entrenar modelo
    model = ARIMAXForecaster(metal_name, order=(1,1,1))
    model.fit(df_train)
    
    # 4. Predecir
    y_pred = model.predict(df_test)
    
    if y_pred is None:
        return None
    
    # 5. Evaluar
    y_true = df_test['Price']
    metrics = model.evaluate(y_true, y_pred)
    
    # 6. Guardar resultados
    results = pd.DataFrame({
        'Date': df_test.index,
        'Real': y_true.values,
        'ARIMAX_Pred': y_pred.values
    })
    
    output_path = Path(f'results/arimax/{metal_name}_predictions.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    
    logger.info(f"\nResultados guardados: {output_path}")
    
    return {
        'metal': metal_name,
        'metrics': metrics,
        'predictions': results,
        'model': model
    }


if __name__ == '__main__':
    # Ejecutar experimentos para todos los metales
    metals = ['cobre', 'oro', 'plata', 'cobalto']
    
    all_results = {}
    
    for metal in metals:
        result = run_arimax_experiment(metal, test_size=24)
        if result:
            all_results[metal] = result
    
    # Resumen comparativo
    logger.info(f"\n{'='*60}")
    logger.info("RESUMEN COMPARATIVO ARIMAX")
    logger.info(f"{'='*60}\n")
    
    summary = []
    for metal, data in all_results.items():
        summary.append({
            'Metal': metal.upper(),
            'MAPE': f"{data['metrics']['MAPE']:.2f}%",
            'MAE': f"${data['metrics']['MAE']:.2f}",
            'RMSE': f"${data['metrics']['RMSE']:.2f}"
        })
    
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    # Guardar resumen
    df_summary.to_csv('results/arimax/summary_arimax.csv', index=False)
