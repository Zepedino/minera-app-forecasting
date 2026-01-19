"""
Evaluación con análisis de sensibilidad de ratios train/test.
"""
import pandas as pd
from typing import List, Callable
from src.utils import setup_logger, calculate_all_metrics

logger = setup_logger(__name__)


class CVEvaluator:
    """
    Evaluador para análisis de sensibilidad de splits train/test.
    """
    
    @staticmethod
    def sensitivity_analysis(
        data: pd.Series,
        fit_function: Callable,
        test_ratios: List[float] = [0.1, 0.2, 0.3]
    ) -> pd.DataFrame:
        """
        Análisis de sensibilidad con diferentes ratios train/test.
        
        Args:
            data: Serie temporal completa
            fit_function: Función de ajuste (train_data, forecast_steps) -> forecast
            test_ratios: Lista de ratios a probar (default: [0.1, 0.2, 0.3] = 90/10, 80/20, 70/30)
            
        Returns:
            DataFrame con resultados por ratio
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
