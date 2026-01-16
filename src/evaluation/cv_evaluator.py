"""
Evaluación con Cross-Validation temporal.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from src.utils import setup_logger, calculate_all_metrics

logger = setup_logger(__name__)


class CVEvaluator:
    """
    Evaluador para Cross-Validation temporal.
    """
    
    @staticmethod
    def expanding_window_cv(
        data: pd.Series,
        fit_function: Callable,
        min_train_size: int = 200,
        test_ratio: float = 0.2,
        step_size: int = 50,
        min_folds: int = 3
    ) -> Dict:
        """
        Expanding Window Cross-Validation con ratio fijo.
        
        Args:
            data: Serie temporal completa
            fit_function: Función que entrena y predice (train_data, forecast_steps) -> forecast
            min_train_size: Tamaño mínimo de entrenamiento
            test_ratio: Porcentaje de test (0.2 = 20%)
            step_size: Pasos entre folds
            min_folds: Número mínimo de folds requeridos
            
        Returns:
            Dict con resultados de CV
        """
        total_size = len(data)
        folds_results = []
        
        logger.info(f"Expanding Window CV: min_train={min_train_size}, test_ratio={test_ratio}, step={step_size}")
        
        current_size = min_train_size
        fold_num = 1
        
        while current_size <= total_size:
            train_size = int(current_size * (1 - test_ratio))
            test_size = current_size - train_size
            
            if train_size < min_train_size or test_size < 1:
                current_size += step_size
                continue
            
            train_data = data[:train_size]
            test_data = data[train_size:current_size]
            
            if len(test_data) < 1:
                break
            
            logger.info(f"Fold {fold_num}: train={len(train_data)}, test={len(test_data)}")
            
            try:
                forecast = fit_function(train_data, len(test_data))
                
                metrics = calculate_all_metrics(test_data.values, forecast)
                
                folds_results.append({
                    'fold': fold_num,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    **metrics
                })
                
                fold_num += 1
                
            except Exception as e:
                logger.error(f"Error en fold {fold_num}: {e}")
            
            current_size += step_size
            
            if fold_num > 20:
                logger.warning("Máximo 20 folds alcanzado")
                break
        
        if len(folds_results) < min_folds:
            logger.warning(f"Solo {len(folds_results)} folds generados (mínimo: {min_folds})")
        
        df_results = pd.DataFrame(folds_results)
        
        avg_metrics = {
            'mean_MAPE': df_results['MAPE'].mean(),
            'std_MAPE': df_results['MAPE'].std(),
            'mean_MAE': df_results['MAE'].mean(),
            'mean_RMSE': df_results['RMSE'].mean(),
            'n_folds': len(folds_results)
        }
        
        logger.info(f"CV Completado: {len(folds_results)} folds, MAPE medio={avg_metrics['mean_MAPE']:.2f}%")
        
        return {
            'folds': df_results,
            'summary': avg_metrics
        }
    
    @staticmethod
    def sensitivity_analysis(
        data: pd.Series,
        fit_function: Callable,
        test_ratios: List[float] = [0.1, 0.2, 0.3]
    ) -> Dict:
        """
        Análisis de sensibilidad con diferentes ratios train/test.
        
        Args:
            data: Serie temporal completa
            fit_function: Función de ajuste
            test_ratios: Lista de ratios a probar
            
        Returns:
            Dict con resultados por ratio
        """
        results = []
        
        logger.info(f"Análisis de sensibilidad: ratios={test_ratios}")
        
        for ratio in test_ratios:
            train_size = int(len(data) * (1 - ratio))
            test_size = len(data) - train_size
            
            train = data[:train_size]
            test = data[train_size:]
            
            logger.info(f"Ratio {int((1-ratio)*100)}/{int(ratio*100)}: train={len(train)}, test={len(test)}")
            
            try:
                forecast = fit_function(train, len(test))
                
                metrics = calculate_all_metrics(test.values, forecast)
                
                results.append({
                    'train_ratio': 1 - ratio,
                    'test_ratio': ratio,
                    'train_size': len(train),
                    'test_size': len(test),
                    **metrics
                })
                
            except Exception as e:
                logger.error(f"Error en ratio {ratio}: {e}")
        
        df_results = pd.DataFrame(results)
        
        logger.info("Análisis de sensibilidad completado")
        
        return df_results
