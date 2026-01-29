"""
Optimizador de variables exógenas para ARIMAX
Implementa Forward Selection para elegir mejores variables
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger
from src.models.arimax_model import ARIMAXForecaster

logger = setup_logger(__name__)


class ARIMAXOptimizer:
    """
    Optimiza selección de variables exógenas mediante Forward Selection
    """
    
    def __init__(self, metal_name, test_ratio=0.2):
        self.metal_name = metal_name
        self.test_ratio = test_ratio
        self.results = []
        
    def load_data(self):
        """Carga datos procesados con todas las exógenas disponibles"""
        data_path = Path(f'data/processed/{self.metal_name}_with_exogenous.csv')
        
        if not data_path.exists():
            logger.error(f"Datos no encontrados: {data_path}")
            return None, None, None
        
        df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        
        test_size = int(len(df) * self.test_ratio)
        split_idx = len(df) - test_size
        
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        
        available_exog = [col for col in df.columns if col != 'Price']
        
        logger.info(f"\nDatos cargados para {self.metal_name.upper()}")
        logger.info(f"  Variables exógenas disponibles: {available_exog}")
        logger.info(f"  Train: {len(df_train)} obs, Test: {len(df_test)} obs")
        
        return df_train, df_test, available_exog
    
    def evaluate_model(self, df_train, df_test, exog_cols=None):
        """
        Entrena y evalúa ARIMAX con variables exógenas específicas
        
        Args:
            df_train: Datos de entrenamiento
            df_test: Datos de test
            exog_cols: Lista de variables exógenas (None = sin exógenas)
            
        Returns:
            dict con métricas
        """
        try:
            model = ARIMAXForecaster(self.metal_name, order=(1,1,1))
            model.fit(df_train, exog_cols=exog_cols)
            
            y_pred = model.predict(df_test)
            y_true = df_test['Price']
            
            metrics = model.evaluate(y_true, y_pred)
            
            return {
                'exog_vars': exog_cols if exog_cols else [],
                'n_vars': len(exog_cols) if exog_cols else 0,
                'MAPE': metrics['MAPE'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'AIC': model.results.aic,
                'BIC': model.results.bic
            }
            
        except Exception as e:
            logger.error(f"Error evaluando modelo con {exog_cols}: {e}")
            return None
    
    def forward_selection(self, df_train, df_test, available_exog, max_vars=None):
        """
        Forward Selection: agrega variables una a una si mejoran MAPE
        
        Args:
            df_train: Datos de entrenamiento
            df_test: Datos de test
            available_exog: Lista de variables exógenas disponibles
            max_vars: Máximo de variables a seleccionar (None = sin límite)
            
        Returns:
            Lista de variables seleccionadas
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"FORWARD SELECTION: {self.metal_name.upper()}")
        logger.info(f"{'='*60}")
        
        logger.info("\n[PASO 0] Modelo baseline (sin exógenas)")
        baseline = self.evaluate_model(df_train, df_test, exog_cols=None)
        
        if baseline is None:
            logger.error("Error en modelo baseline")
            return []
        
        self.results.append(baseline)
        logger.info(f"  MAPE baseline: {baseline['MAPE']:.2f}%")
        logger.info(f"  AIC: {baseline['AIC']:.2f}")
        
        selected_vars = []
        remaining_vars = available_exog.copy()
        best_mape = baseline['MAPE']
        
        step = 1
        while remaining_vars:
            if max_vars and len(selected_vars) >= max_vars:
                logger.info(f"\nLímite de {max_vars} variables alcanzado")
                break
            
            logger.info(f"\n[PASO {step}] Probando agregar variables...")
            
            candidates = []
            
            for var in remaining_vars:
                test_vars = selected_vars + [var]
                logger.info(f"  Probando: {test_vars}")
                
                result = self.evaluate_model(df_train, df_test, exog_cols=test_vars)
                
                if result:
                    candidates.append((var, result))
                    logger.info(f"    MAPE: {result['MAPE']:.2f}%, AIC: {result['AIC']:.2f}")
            
            if not candidates:
                logger.warning("  No hay candidatos válidos")
                break
            
            candidates.sort(key=lambda x: x[1]['MAPE'])
            best_var, best_result = candidates[0]
            
            improvement = best_mape - best_result['MAPE']
            
            logger.info(f"\n  Mejor variable: {best_var}")
            logger.info(f"  MAPE: {best_result['MAPE']:.2f}% (mejora: {improvement:.2f}pp)")
            
            if best_result['MAPE'] < best_mape:
                selected_vars.append(best_var)
                remaining_vars.remove(best_var)
                best_mape = best_result['MAPE']
                self.results.append(best_result)
                
                logger.info(f"  [OK] Variable agregada. Seleccionadas: {selected_vars}")
            else:
                logger.info(f"  [STOP] Agregar {best_var} no mejora MAPE. Fin de selección.")
                break
            
            step += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SELECCIÓN FINAL: {selected_vars}")
        logger.info(f"MAPE final: {best_mape:.2f}% (mejora: {baseline['MAPE'] - best_mape:.2f}pp)")
        logger.info(f"{'='*60}")
        
        return selected_vars
    
    def exhaustive_search(self, df_train, df_test, available_exog, max_vars=3):
        """
        Prueba todas las combinaciones de variables (hasta max_vars)
        
        Args:
            df_train: Datos de entrenamiento
            df_test: Datos de test
            available_exog: Lista de variables exógenas disponibles
            max_vars: Máximo de variables por combinación
            
        Returns:
            Lista de variables seleccionadas
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"EXHAUSTIVE SEARCH: {self.metal_name.upper()}")
        logger.info(f"{'='*60}")
        
        logger.info("\n[BASELINE] Sin exógenas")
        baseline = self.evaluate_model(df_train, df_test, exog_cols=None)
        self.results.append(baseline)
        logger.info(f"  MAPE: {baseline['MAPE']:.2f}%")
        
        best_result = baseline
        best_vars = []
        
        total_combinations = sum(
            len(list(combinations(available_exog, r))) 
            for r in range(1, min(max_vars + 1, len(available_exog) + 1))
        )
        
        logger.info(f"\nProbando {total_combinations} combinaciones...")
        
        tested = 0
        for n_vars in range(1, min(max_vars + 1, len(available_exog) + 1)):
            for combo in combinations(available_exog, n_vars):
                tested += 1
                exog_list = list(combo)
                
                logger.info(f"[{tested}/{total_combinations}] Probando: {exog_list}")
                
                result = self.evaluate_model(df_train, df_test, exog_cols=exog_list)
                
                if result:
                    self.results.append(result)
                    logger.info(f"  MAPE: {result['MAPE']:.2f}%, AIC: {result['AIC']:.2f}")
                    
                    if result['MAPE'] < best_result['MAPE']:
                        best_result = result
                        best_vars = exog_list
                        logger.info(f"  [MEJOR] Nueva mejor combinación")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MEJOR COMBINACIÓN: {best_vars}")
        logger.info(f"MAPE: {best_result['MAPE']:.2f}%")
        logger.info(f"Mejora vs baseline: {baseline['MAPE'] - best_result['MAPE']:.2f}pp")
        logger.info(f"{'='*60}")
        
        return best_vars


def optimize_all_metals(method='forward', max_vars=3):
    """
    Optimiza variables exógenas para todos los metales
    
    Args:
        method: 'forward' o 'exhaustive'
        max_vars: Máximo de variables a seleccionar
    """
    metals = ['cobre', 'oro', 'plata', 'cobalto']
    
    optimization_results = {}
    
    for metal in metals:
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"OPTIMIZANDO: {metal.upper()}")
        logger.info(f"{'#'*60}")
        
        optimizer = ARIMAXOptimizer(metal, test_ratio=0.2)
        
        df_train, df_test, available_exog = optimizer.load_data()
        
        if df_train is None:
            logger.error(f"No se pudieron cargar datos para {metal}")
            continue
        
        if method == 'forward':
            selected_vars = optimizer.forward_selection(
                df_train, df_test, available_exog, max_vars=max_vars
            )
        elif method == 'exhaustive':
            selected_vars = optimizer.exhaustive_search(
                df_train, df_test, available_exog, max_vars=max_vars
            )
        else:
            logger.error(f"Método desconocido: {method}")
            continue
        
        optimization_results[metal] = {
            'selected_vars': selected_vars,
            'all_results': optimizer.results
        }
    
    output_dir = Path('results/optimization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = []
    for metal, data in optimization_results.items():
        if data['all_results']:
            best = min(data['all_results'], key=lambda x: x['MAPE'])
            baseline = next(r for r in data['all_results'] if r['n_vars'] == 0)
            
            summary.append({
                'Metal': metal.upper(),
                'Variables_Seleccionadas': ', '.join(data['selected_vars']),
                'N_Variables': len(data['selected_vars']),
                'MAPE_Baseline': f"{baseline['MAPE']:.2f}%",
                'MAPE_Optimizado': f"{best['MAPE']:.2f}%",
                'Mejora': f"{baseline['MAPE'] - best['MAPE']:.2f}pp"
            })
            
            df_detail = pd.DataFrame(data['all_results'])
            df_detail['Metal'] = metal
            df_detail.to_csv(output_dir / f'{metal}_optimization_detail.csv', index=False)
    
    df_summary = pd.DataFrame(summary)
    print("\n" + "="*80)
    print("RESUMEN DE OPTIMIZACIÓN")
    print("="*80)
    print(df_summary.to_string(index=False))
    
    df_summary.to_csv(output_dir / 'optimization_summary.csv', index=False)
    logger.info(f"\nResumen guardado: {output_dir / 'optimization_summary.csv'}")
    
    optimal_config = {}
    for metal, data in optimization_results.items():
        optimal_config[metal] = data['selected_vars']
    
    logger.info(f"\n{'='*60}")
    logger.info("CONFIGURACIÓN ÓPTIMA PARA merge_exogenous.py:")
    logger.info(f"{'='*60}")
    print("\nEXOGENOUS_CONFIG = {")
    for metal, vars_list in optimal_config.items():
        print(f"    '{metal}': {vars_list},")
    print("}")


if __name__ == '__main__':
    optimize_all_metals(method='forward', max_vars=3)
