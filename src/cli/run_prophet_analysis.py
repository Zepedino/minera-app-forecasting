"""
Analisis de Prophet para commodities.

Compara configuracion base, conservadora y flexible usando las mismas
metricas (MAPE, MAE, RMSE) y esquema de train/test que Holt y ARIMA.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils import (
    setup_logger,
    load_and_validate,
    calculate_all_metrics,
)
from src.models import (
    fit_prophet,
    fit_prophet_conservative,
    fit_prophet_flexible,
)
from config import TRAIN_TEST_SPLIT

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analisis Prophet (base / conservador / flexible)"
    )

    parser.add_argument(
        "--commodity",
        type=str,
        required=True,
        choices=["oro", "plata", "cobre", "cobalto"],
        help="Commodity a analizar",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TRAIN_TEST_SPLIT["test_ratio"],
        help="Ratio de test (default: 0.2)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/prophet",
        help="Directorio de salida",
    )

    return parser.parse_args()


def evaluate_prophet_variant(
    data,
    fit_func,
    model_name: str,
    test_ratio: float,
):
    """
    Ajusta y evalua una variante de Prophet.

    Devuelve un diccionario con modelo, tamaños de train/test y métricas.
    """
    n_obs = len(data)
    train_size = int(n_obs * (1 - test_ratio))
    train = data[:train_size]
    test = data[train_size:]

    logger.info(f"\n{model_name}: train={len(train)}, test={len(test)}")

    try:
        # Ajuste del modelo
        forecast, model = fit_func(train, len(test))

        # Calculo de métricas
        metrics = calculate_all_metrics(test.values, forecast)

        result = {
            "model": model_name,
            "train_size": len(train),
            "test_size": len(test),
            **metrics,
        }

        logger.info(
            f"{model_name} - MAPE: {metrics['MAPE']:.2f}% | "
            f"MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f}"
        )

        return result

    except Exception as e:
        logger.error(f"Error en {model_name}: {e}", exc_info=True)
        return None


def run_prophet_analysis(commodity: str, test_ratio: float, output_dir: str):
    logger.info("=" * 70)
    logger.info(f"ANALISIS PROPHET - {commodity.upper()}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # 1) Cargar datos                                                   #
    # ------------------------------------------------------------------ #
    logger.info(f"\nCargando datos de {commodity}...")
    data = load_and_validate(commodity)
    logger.info(
        f"Datos: {len(data)} obs ({data.index[0].date()} a {data.index[-1].date()})"
    )

    # Directorio de salida
    output_path = Path(output_dir) / commodity
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 2) Evaluar variantes de Prophet                                   #
    # ------------------------------------------------------------------ #
    results = []

    models = [
        ("Prophet_Base", fit_prophet),
        ("Prophet_Conservative", fit_prophet_conservative),
        ("Prophet_Flexible", fit_prophet_flexible),
    ]

    for model_name, func in models:
        res = evaluate_prophet_variant(
            data=data,
            fit_func=func,
            model_name=model_name,
            test_ratio=test_ratio,
        )
        if res is not None:
            results.append(res)

    if not results:
        logger.error("No se pudo evaluar ninguna variante de Prophet.")
        return

    # ------------------------------------------------------------------ #
    # 3) Guardar resultados y resumen                                   #
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame(results)
    results_file = output_path / "prophet_comparison.csv"
    results_df.to_csv(results_file, index=False)

    logger.info(f"\nResultados guardados en: {results_file}")

    # Tabla resumen ordenada por MAPE
    logger.info("\nCOMPARACION FINAL (ordenado por MAPE)")
    comparison = results_df[["model", "MAPE", "MAE", "RMSE"]].sort_values("MAPE")
    logger.info("\n" + comparison.to_string(index=False))

    best = comparison.iloc[0]
    logger.info(
        f"\nMEJOR CONFIGURACION PROPHET: {best['model']} "
        f"(MAPE={best['MAPE']:.2f}%, MAE={best['MAE']:.2f}, RMSE={best['RMSE']:.2f})"
    )


def main():
    args = parse_args()

    try:
        run_prophet_analysis(
            commodity=args.commodity,
            test_ratio=args.test_ratio,
            output_dir=args.output_dir,
        )
        logger.info("\n✓ Analisis Prophet completado exitosamente")
    except Exception as e:
        logger.error(f"\n✗ Error en analisis Prophet: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
