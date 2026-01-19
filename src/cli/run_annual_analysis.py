"""
Análisis de modelos baseline con datos anuales (resample de mensuales).
Compara performance de forecasting mensual vs anual.
"""
import pandas as pd
import argparse
from pathlib import Path
from src.utils import setup_logger, load_and_validate
from src.models.arima import fit_arima
from src.models.exponential_smoothing import fit_holt, fit_damped

logger = setup_logger(__name__)


def resample_to_annual(data: pd.Series) -> pd.Series:
    """
    Convertir serie mensual a anual usando promedio.
    
    Args:
        data: Serie temporal mensual
        
    Returns:
        Serie temporal anual (promedio por año)
    """
    # Asegurar que el índice sea datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Resample a anual usando promedio
    annual_data = data.resample('YE').mean()  # YE = Year End
    
    logger.info(f"Resampled: {len(data)} obs mensuales → {len(annual_data)} obs anuales")
    
    return annual_data


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Análisis con datos anuales (resample de mensuales)'
    )
    
    parser.add_argument(
        '--commodity',
        type=str,
        required=True,
        choices=['oro', 'plata', 'cobre', 'cobalto'],
        help='Commodity a analizar'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Ratio de datos para test (default: 0.2 = 80/20)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/annual',
        help='Directorio para guardar resultados'
    )
    
    return parser.parse_args()


def create_fit_function(fit_func, model_name: str):
    """
    Crear función de fit compatible con evaluación.
    
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
            import numpy as np
            return np.array([train_data.iloc[-1]] * steps)
    
    return fit_and_forecast


def evaluate_model(data: pd.Series, fit_func, model_name: str, test_ratio: float):
    """
    Evaluar un modelo con split train/test.
    
    Args:
        data: Serie temporal
        fit_func: Función de ajuste
        model_name: Nombre del modelo
        test_ratio: Proporción de datos para test
        
    Returns:
        Dict con resultados
    """
    from src.utils import calculate_all_metrics
    
    # Split train/test
    train_size = int(len(data) * (1 - test_ratio))
    train = data[:train_size]
    test = data[train_size:]
    
    logger.info(f"{model_name}: train={len(train)}, test={len(test)}")
    
    # Ajustar y predecir
    fit_wrapper = create_fit_function(fit_func, model_name)
    forecast = fit_wrapper(train, len(test))
    
    # Calcular métricas
    metrics = calculate_all_metrics(test.values, forecast)
    
    return {
        'model': model_name,
        'train_size': len(train),
        'test_size': len(test),
        'train_start': train.index[0],
        'train_end': train.index[-1],
        'test_start': test.index[0],
        'test_end': test.index[-1],
        **metrics
    }


def run_annual_analysis(commodity: str, test_ratio: float, output_dir: str):
    """
    Ejecutar análisis con datos anuales.
    
    Args:
        commodity: Nombre del commodity
        test_ratio: Proporción de test
        output_dir: Directorio de salida
    """
    logger.info("="*60)
    logger.info(f"ANALISIS CON DATOS ANUALES - {commodity.upper()}")
    logger.info("="*60)
    
    # Cargar datos mensuales
    logger.info("Cargando datos mensuales...")
    monthly_data = load_and_validate(commodity)
    logger.info(f"Datos mensuales: {len(monthly_data)} obs ({monthly_data.index[0]} a {monthly_data.index[-1]})")
    
    # Convertir a anual
    logger.info("\nConvirtiendo a datos anuales (promedio)...")
    annual_data = resample_to_annual(monthly_data)
    logger.info(f"Datos anuales: {len(annual_data)} obs ({annual_data.index[0].year} a {annual_data.index[-1].year})")
    
    # Preparar directorio de salida
    output_path = Path(output_dir) / commodity
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Guardar datos anuales
    annual_data.to_csv(output_path / f"{commodity}_annual_data.csv", header=['Price'])
    logger.info(f"\nDatos anuales guardados en: {output_path / f'{commodity}_annual_data.csv'}")
    
    # Modelos a evaluar
    models = {
        'ARIMA': fit_arima,
        'Holt_Linear': fit_holt,
        'Damped_Trend': fit_damped
    }
    
    results = []
    
    # Evaluar cada modelo
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUACIÓN CON SPLIT {int((1-test_ratio)*100)}/{int(test_ratio*100)}")
    logger.info(f"{'='*60}\n")
    
    for model_name, fit_func in models.items():
        logger.info(f"Evaluando {model_name}...")
        
        result = evaluate_model(annual_data, fit_func, model_name, test_ratio)
        results.append(result)
        
        logger.info(f"  MAPE: {result['MAPE']:.2f}%")
        logger.info(f"  MAE:  {result['MAE']:.2f}")
        logger.info(f"  RMSE: {result['RMSE']:.2f}\n")
    
    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "annual_results.csv", index=False)
    
    # Comparación
    logger.info(f"{'='*60}")
    logger.info("COMPARACIÓN DE MODELOS")
    logger.info(f"{'='*60}\n")
    
    comparison = results_df[['model', 'MAPE', 'MAE', 'RMSE']].sort_values('MAPE')
    logger.info(comparison.to_string(index=False))
    
    best_model = comparison.iloc[0]
    logger.info(f"\nMEJOR MODELO: {best_model['model']} (MAPE: {best_model['MAPE']:.2f}%)")
    
    logger.info(f"\nResultados guardados en: {output_path}")


def main():
    """Función principal."""
    args = parse_args()
    
    try:
        run_annual_analysis(
            commodity=args.commodity,
            test_ratio=args.test_ratio,
            output_dir=args.output_dir
        )
        
        logger.info("\nAnálisis completado exitosamente")
        
    except Exception as e:
        logger.error(f"\nError en análisis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
