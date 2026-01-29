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
        self.exog_cols = None
        
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
        
        y_train = df_train['Price'].copy()
        
        if y_train.isnull().any():
            n_nulls = y_train.isnull().sum()
            logger.warning(f"  {n_nulls} valores nulos en Price - rellenando")
            y_train = y_train.fillna(method='ffill').fillna(method='bfill')
        
        if exog_cols is None:
            exog_cols = [col for col in df_train.columns if col != 'Price']
        
        self.exog_cols = exog_cols
        
        if len(exog_cols) == 0:
            logger.warning("  Sin variables exógenas, usando ARIMA puro")
            X_train = None
        else:
            X_train = df_train[exog_cols].copy()
            
            if X_train.isnull().any().any():
                nulls_per_col = X_train.isnull().sum()
                logger.warning(f"  Valores nulos en exógenas:")
                for col, n in nulls_per_col[nulls_per_col > 0].items():
                    logger.warning(f"    {col}: {n}")
                X_train = X_train.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"  Variables exógenas: {exog_cols}")
            logger.info(f"  Shape: {X_train.shape}")
        
        logger.info(f"  Observaciones: {len(y_train)}")
        logger.info(f"  Orden ARIMA: {self.order}")
        
        logger.info("\n  Test de Estacionariedad:")
        self.check_stationarity(y_train)
        
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
            raise
    
    def predict(self, df_test):
        """
        Genera predicciones
        
        Args:
            df_test: DataFrame con variables exógenas para forecast
        
        Returns:
            Series con predicciones
        """
        if self.results is None:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero")
        
        if self.exog_cols is not None and len(self.exog_cols) > 0:
            missing = set(self.exog_cols) - set(df_test.columns)
            if missing:
                raise ValueError(f"Columnas faltantes en test: {missing}")
            
            X_test = df_test[self.exog_cols].copy()
            
            if X_test.isnull().any().any():
                logger.warning("  NaN en test - rellenando")
                X_test = X_test.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"  Prediciendo con exógenas: {self.exog_cols}")
        else:
            X_test = None
            logger.info(f"  Prediciendo sin exógenas")
        
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
            raise
    
    def evaluate(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        mask = ~(y_true.isnull() | y_pred.isnull())
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
        
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


def run_arimax_experiment(metal_name, test_ratio=0.2):
    """
    Ejecuta experimento completo ARIMAX para un metal
    
    Args:
        metal_name: 'cobre', 'oro', 'plata', 'cobalto'
        test_ratio: Proporción para test (default 0.2 = 20%)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENTO ARIMAX: {metal_name.upper()}")
    logger.info(f"{'='*60}")
    
    data_path = Path(f'data/processed/{metal_name}_with_exogenous.csv')
    
    if not data_path.exists():
        logger.error(f"Datos no encontrados: {data_path}")
        logger.error(f"Ejecuta: python src/cli/run_arimax.py")
        return None
    
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    logger.info(f"\nDatos cargados: {df.shape}")
    logger.info(f"Columnas: {list(df.columns)}")
    logger.info(f"Rango: {df.index[0]} a {df.index[-1]}")
    
    test_size = int(len(df) * test_ratio)
    split_idx = len(df) - test_size
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    train_pct = (len(df_train) / len(df)) * 100
    test_pct = (len(df_test) / len(df)) * 100
    
    logger.info(f"\nSplit:")
    logger.info(f"  Train: {len(df_train)} obs ({train_pct:.1f}%) - {df_train.index[0]} a {df_train.index[-1]}")
    logger.info(f"  Test:  {len(df_test)} obs ({test_pct:.1f}%) - {df_test.index[0]} a {df_test.index[-1]}")
    
    if len(df_train) < 50:
        logger.error(f"Datos insuficientes para entrenar (mínimo 50 obs)")
        return None
    
    try:
        model = ARIMAXForecaster(metal_name, order=(1,1,1))
        model.fit(df_train)
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        return None
    
    try:
        y_pred = model.predict(df_test)
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return None
    
    y_true = df_test['Price']
    metrics = model.evaluate(y_true, y_pred)
    
    results = pd.DataFrame({
        'Date': df_test.index,
        'Real': y_true.values,
        'ARIMAX_Pred': y_pred.values
    })
    
    output_path = Path(f'results/arimax/{metal_name}_predictions.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    
    logger.info(f"\nResultados guardados: {output_path}")
    
    results_with_metrics = results.copy()
    results_with_metrics['Error_Abs'] = abs(results['Real'] - results['ARIMAX_Pred'])
    results_with_metrics['Error_Pct'] = abs((results['Real'] - results['ARIMAX_Pred']) / results['Real']) * 100
    
    metrics_path = Path(f'results/arimax/{metal_name}_metrics_detailed.csv')
    results_with_metrics.to_csv(metrics_path, index=False)
    
    logger.info(f"Métricas detalladas guardadas: {metrics_path}")
    
    return {
        'metal': metal_name,
        'metrics': metrics,
        'predictions': results,
        'model': model
    }


if __name__ == '__main__':
    metals = ['cobre', 'oro', 'plata', 'cobalto']
    
    all_results = {}
    
    for metal in metals:
        try:
            result = run_arimax_experiment(metal, test_ratio=0.2)
            if result:
                all_results[metal] = result
        except Exception as e:
            logger.error(f"Error procesando {metal}: {e}")
            continue
    
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("RESUMEN COMPARATIVO ARIMAX (80/20 split)")
        logger.info(f"{'='*60}\n")
        
        summary = []
        for metal, data in all_results.items():
            df_pred = data['predictions']
            n_test = len(df_pred)
            
            summary.append({
                'Metal': metal.upper(),
                'N_Test': n_test,
                'MAPE': f"{data['metrics']['MAPE']:.2f}%",
                'MAE': f"${data['metrics']['MAE']:.2f}",
                'RMSE': f"${data['metrics']['RMSE']:.2f}",
                'Precio_Real_Promedio': f"${df_pred['Real'].mean():.2f}",
                'Precio_Pred_Promedio': f"${df_pred['ARIMAX_Pred'].mean():.2f}"
            })
        
        df_summary = pd.DataFrame(summary)
        print("\n" + df_summary.to_string(index=False))
        
        output_dir = Path('results/arimax')
        output_dir.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(output_dir / 'summary_arimax.csv', index=False)
        logger.info(f"\nResumen guardado: {output_dir / 'summary_arimax.csv'}")
        
        summary_raw = []
        for metal, data in all_results.items():
            summary_raw.append({
                'Metal': metal,
                'MAPE': data['metrics']['MAPE'],
                'MAE': data['metrics']['MAE'],
                'RMSE': data['metrics']['RMSE'],
                'N_Test': len(data['predictions'])
            })
        
        df_summary_raw = pd.DataFrame(summary_raw)
        df_summary_raw.to_csv(output_dir / 'summary_arimax_raw.csv', index=False)
        logger.info(f"Métricas RAW guardadas: {output_dir / 'summary_arimax_raw.csv'}")
    else:
        logger.error("No se generaron resultados")
