"""
Script para análisis de sensibilidad de ratios train/test.
Evalúa modelos baseline con splits 70/30, 80/20, 90/10.
"""
import pandas as pd
import argparse
from pathlib import Path
from src.utils import setup_logger, load_and_validate
from src.evaluation import CVEvaluator
from src.models.arima import fit_arima
from src.models.exponential_smoothing import fit_holt, fit_damped

logger = setup_logger(__name__)


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Análisis de sensibilidad de ratios train/test'
    )
    
    parser.add_argument(
        '--commodity',
        type=str,
        required=True,
        choices=['oro', 'plata', 'cobre', 'cobalto'],
        help='Commodity a analizar'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/sensitivity',
        help='Directorio para guardar resultados'
    )
    
    return parser.parse_args()


def create_fit_function(fit_func, model_name: str):
    """
    Crear función de fit compatible con CVEvaluator.
    
    Args:
        fit_func: Función del modelo (fit_arima, fit_holt, etc.)
        model_name: Nombre del modelo para logging
        
    Returns:
        Función que acepta (train_data, steps) y retorna forecast
    """
    def fit_and_forecast(train_data: pd.Series, steps: int):
        """Ajustar modelo y generar pronóstico."""
        try:
            forecast, _ = fit_func(train_data, steps)
            return forecast.values if isinstance(forecast, pd.Series) else forecast
        except Exception as e:
            logger.error(f"Error en {model_name}: {e}")
            # Retornar predicción naive como fallback
            import numpy as np
            return np.array([train_data.iloc[-1]] * steps)
    
    return fit_and_forecast


def run_sensitivity_analysis(commodity: str, output_dir: str):
    """
    Ejecutar análisis de sensibilidad para un commodity.
    
    Args:
        commodity: Nombre del commodity
        output_dir: Directorio de salida
    """
    logger.info("="*60)
    logger.info(f"ANALISIS DE SENSIBILIDAD - {commodity.upper()}")
    logger.info("="*60)
    
    # Cargar datos
    logger.info(f"Cargando datos de {commodity}...")
    data = load_and_validate(commodity)
    logger.info(f"Datos cargados: {len(data)} observaciones")
    logger.info(f"Período: {data.index[0]} a {data.index[-1]}")
    
    # Preparar directorio de salida
    output_path = Path(output_dir) / commodity
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ratios a evaluar: 90/10, 80/20, 70/30
    test_ratios = [0.1, 0.2, 0.3]
    
    # Modelos a evaluar
    models = {
        'ARIMA': fit_arima,
        'Holt_Linear': fit_holt,
        'Damped_Trend': fit_damped
    }
    
    all_results = []
    
    # Evaluar cada modelo
    for model_name, fit_func in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluando {model_name}")
        logger.info(f"{'='*60}")
        
        # Crear función wrapper
        fit_wrapper = create_fit_function(fit_func, model_name)
        
        # Ejecutar análisis de sensibilidad
        results_df = CVEvaluator.sensitivity_analysis(
            data=data,
            fit_function=fit_wrapper,
            test_ratios=test_ratios
        )
        
        # Agregar nombre del modelo
        results_df['model'] = model_name
        
        # Guardar resultados por modelo
        results_df.to_csv(
            output_path / f"{model_name}_sensitivity.csv",
            index=False
        )
        
        # Log resultados
        logger.info(f"\n{model_name} - Resultados:")
        for _, row in results_df.iterrows():
            logger.info(f"  Ratio {int(row['train_ratio']*100)}/{int(row['test_ratio']*100)}: "
                       f"MAPE={row['MAPE']:.2f}%, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")
        
        all_results.append(results_df)
    
    # Consolidar resultados
    consolidated = pd.concat(all_results, ignore_index=True)
    consolidated = consolidated[['model', 'train_ratio', 'test_ratio', 
                                 'train_size', 'test_size', 'MAPE', 'MAE', 'RMSE', 'SMAPE']]
    
    consolidated.to_csv(
        output_path / "consolidated_sensitivity.csv",
        index=False
    )
    
    # Mostrar comparación
    logger.info(f"\n{'='*60}")
    logger.info("RESULTADOS CONSOLIDADOS")
    logger.info(f"{'='*60}\n")
    
    # Pivot table para mejor visualización
    pivot = consolidated.pivot_table(
        values='MAPE',
        index='model',
        columns='test_ratio',
        aggfunc='first'
    )
    pivot.columns = [f"{int((1-x)*100)}/{int(x*100)}" for x in pivot.columns]
    
    logger.info("MAPE por modelo y ratio (train/test):")
    logger.info(f"\n{pivot.to_string()}\n")
    
    # Mejor configuración por modelo
    logger.info("\nMejor ratio por modelo:")
    for model_name in models.keys():
        model_data = consolidated[consolidated['model'] == model_name]
        best_idx = model_data['MAPE'].idxmin()
        best_row = model_data.loc[best_idx]
        logger.info(f"  {model_name}: {int(best_row['train_ratio']*100)}/{int(best_row['test_ratio']*100)} "
                   f"(MAPE: {best_row['MAPE']:.2f}%)")
    
    logger.info(f"\nResultados guardados en: {output_path}")


def main():
    """Función principal."""
    args = parse_args()
    
    try:
        run_sensitivity_analysis(
            commodity=args.commodity,
            output_dir=args.output_dir
        )
        
        logger.info("\nAnalisis completado exitosamente")
        
    except Exception as e:
        logger.error(f"\nError en análisis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
