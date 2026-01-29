"""
Combina datos de metales con variables exógenas
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

logger = setup_logger(__name__)


# Configuración de variables exógenas por metal
EXOGENOUS_CONFIG = {
    'cobre': ['USD_Index', 'Brent_Oil'],
    'oro': ['USD_Index', 'VIX'],
    'plata': ['USD_Index', 'VIX', 'Brent_Oil'],
    'cobalto': ['USD_Index', 'Brent_Oil']
}


def load_exogenous_variable(var_name):
    """Carga una variable exógena limpia"""
    filepath = Path(f'data/exogenous/clean/{var_name}.csv')
    
    if not filepath.exists():
        logger.error(f"Variable {var_name} no encontrada en {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    return df


def merge_metal_with_exogenous(metal_name, start_date='1990-01-01', end_date='2025-12-31'):
    """
    Combina datos de un metal con sus variables exógenas configuradas.
    
    Args:
        metal_name: 'cobre', 'oro', 'plata', 'cobalto'
        start_date: Fecha inicio del dataset
        end_date: Fecha fin del dataset
    
    Returns:
        DataFrame con [Price, var1, var2, ...]
    """
    logger.info(f"=== Procesando {metal_name.upper()} ===")
    
    # 1. Cargar datos del metal
    metal_file = Path(f'data/metals/{metal_name}_monthly.csv')
    
    if not metal_file.exists():
        logger.error(f"Archivo de metal no encontrado: {metal_file}")
        return None
    
    df_metal = pd.read_csv(metal_file)
    df_metal['Date'] = pd.to_datetime(df_metal['Date'])
    df_metal = df_metal.set_index('Date')
    df_metal = df_metal[['Price']].copy()
    
    logger.info(f"  Metal cargado: {len(df_metal)} obs ({df_metal.index.min()} a {df_metal.index.max()})")
    
    # 2. Cargar variables exógenas configuradas
    exog_vars = EXOGENOUS_CONFIG.get(metal_name, [])
    logger.info(f"  Variables exógenas: {exog_vars}")
    
    df_combined = df_metal.copy()
    
    for var_name in exog_vars:
        df_var = load_exogenous_variable(var_name)
        
        if df_var is None:
            logger.warning(f"  Saltando {var_name} (no encontrada)")
            continue
        
        df_combined = df_combined.join(df_var, how='left')
        logger.info(f"  Agregado {var_name}")
    
    # 3. Filtrar rango de fechas
    df_combined = df_combined.loc[start_date:end_date]
    
    # 4. Validar datos faltantes
    missing = df_combined.isnull().sum()
    if missing.any():
        logger.warning(f"  Valores faltantes detectados:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"    {col}: {count} NaN")
    
    # 5. Interpolación lineal para NaN (solo en exógenas)
    exog_cols = [col for col in df_combined.columns if col != 'Price']
    if len(exog_cols) > 0:
        df_combined[exog_cols] = df_combined[exog_cols].interpolate(method='linear')
    
    # 6. Eliminar filas con NaN en Price
    df_combined = df_combined.dropna(subset=['Price'])
    
    logger.info(f"  Dataset final: {len(df_combined)} observaciones")
    logger.info(f"  Columnas: {list(df_combined.columns)}\n")
    
    return df_combined


def prepare_all_metals():
    """Prepara datasets para todos los metales"""
    metals = ['cobre', 'oro', 'plata', 'cobalto']
    datasets = {}
    
    for metal in metals:
        df = merge_metal_with_exogenous(metal)
        if df is not None:
            datasets[metal] = df
            
            # Guardar dataset combinado
            output_path = Path(f'data/processed/{metal}_with_exogenous.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path)
            logger.info(f"Guardado: {output_path}\n")
    
    return datasets


if __name__ == '__main__':
    datasets = prepare_all_metals()
