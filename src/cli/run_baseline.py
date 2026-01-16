"""
Script CLI para ejecutar modelos baseline.

Entrena Holt, Damped y ARIMA para todos los metales (1990-2025).
Genera reporte comparativo con MAPE.
"""

import pandas as pd
from pathlib import Path
from config import DATA_FILES, TRAIN_TEST_SPLIT, BASELINE_RESULTS_DIR
from src.utils import load_and_validate, calculate_all_metrics, compare_models, setup_logger
from src.models import fit_holt, fit_damped, fit_arima

logger = setup_logger(__name__)


def run_baseline_for_metal(metal_name):
    """
    Ejecuta modelos baseline para un metal especifico.
    
    Args:
        metal_name: Nombre del metal ('cobre', 'oro', 'plata', 'cobalto')
    
    Returns:
        DataFrame con resultados comparativos
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EJECUTANDO BASELINE PARA {metal_name.upper()}")
    logger.info(f"{'='*60}\n")
    
    # Cargar datos
    data = load_and_validate(metal_name)
    
    # Split train/test
    split_ratio = TRAIN_TEST_SPLIT['train_ratio']
    train_size = int(len(data) * split_ratio)
    
    train = data[:train_size]
    test = data[train_size:]
    forecast_steps = len(test)
    
    logger.info(f"Split: {len(train)} train, {len(test)} test ({split_ratio*100:.0f}%/{(1-split_ratio)*100:.0f}%)")
    
    # Diccionario para almacenar resultados
    results = {}
    
    # 1. HOLT
    try:
        logger.info("\n--- Ejecutando Holt ---")
        forecast_holt, _ = fit_holt(train, forecast_steps)
        results['Holt'] = {'y_true': test.values, 'y_pred': forecast_holt}
    except Exception as e:
        logger.error(f"Holt fallo: {e}")
    
    # 2. DAMPED
    try:
        logger.info("\n--- Ejecutando Damped ---")
        forecast_damped, _ = fit_damped(train, forecast_steps)
        results['Damped'] = {'y_true': test.values, 'y_pred': forecast_damped}
    except Exception as e:
        logger.error(f"Damped fallo: {e}")
    
    # 3. ARIMA
    try:
        logger.info("\n--- Ejecutando ARIMA ---")
        forecast_arima, _ = fit_arima(train, forecast_steps)
        results['ARIMA'] = {'y_true': test.values, 'y_pred': forecast_arima}
    except Exception as e:
        logger.error(f"ARIMA fallo: {e}")
    
    # Comparar modelos
    comparison_df = compare_models(results)
    
    # Guardar resultados
    output_file = BASELINE_RESULTS_DIR / f"{metal_name}_baseline_results.csv"
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"\nResultados guardados en: {output_file}")
    
    return comparison_df


def run_all_metals():
    """
    Ejecuta baseline para todos los metales y genera reporte consolidado.
    """
    logger.info(f"\n{'#'*60}")
    logger.info("INICIANDO ANALISIS BASELINE - TODOS LOS METALES")
    logger.info(f"{'#'*60}\n")
    
    all_results = []
    
    for metal in DATA_FILES.keys():
        try:
            comparison_df = run_baseline_for_metal(metal)
            comparison_df['Metal'] = metal.capitalize()
            all_results.append(comparison_df)
        except Exception as e:
            logger.error(f"Error procesando {metal}: {e}")
            continue
    
    # Consolidar resultados
    if all_results:
        consolidated = pd.concat(all_results, ignore_index=True)
        consolidated = consolidated[['Metal', 'Model', 'MAPE', 'MAE', 'RMSE', 'SMAPE']]
        
        output_file = BASELINE_RESULTS_DIR / "consolidated_baseline_results.csv"
        consolidated.to_csv(output_file, index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("RESULTADOS CONSOLIDADOS")
        logger.info(f"{'='*60}\n")
        logger.info(f"\n{consolidated.to_string(index=False)}\n")
        logger.info(f"Reporte consolidado guardado en: {output_file}")
    
    logger.info(f"\n{'#'*60}")
    logger.info("ANALISIS BASELINE COMPLETADO")
    logger.info(f"{'#'*60}\n")


if __name__ == "__main__":
    run_all_metals()
