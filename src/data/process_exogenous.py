"""
Procesamiento y limpieza de variables exógenas desde investing.com
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

logger = setup_logger(__name__)


def clean_investing_data(filepath, variable_name):
    """
    Limpia archivos CSV de investing.com al formato estándar.
    
    Args:
        filepath: Ruta al archivo CSV crudo
        variable_name: Nombre de la variable (e.g., 'USD_Index')
    
    Returns:
        DataFrame con columnas ['Date', variable_name]
    """
    try:
        # Leer archivo
        df = pd.read_csv(filepath)
        
        # Convertir fecha (formato investing.com: DD.MM.YYYY)
        df['Date'] = pd.to_datetime(df['Fecha'], format='%d.%m.%Y')
        
        # Limpiar precio (quitar comas europeas, convertir a float)
        # Investing.com usa coma como decimal y punto como separador de miles
        precio_limpio = df['Último'].str.replace('.', '', regex=False)  # Quitar separador miles
        precio_limpio = precio_limpio.str.replace(',', '.', regex=False)  # Coma a punto decimal
        
        df[variable_name] = pd.to_numeric(precio_limpio, errors='coerce')
        
        # Ordenar cronológicamente (investing.com viene descendente)
        df = df.sort_values('Date')
        
        # Detectar valores faltantes
        nan_count = df[variable_name].isna().sum()
        if nan_count > 0:
            logger.warning(f"{variable_name}: {nan_count} valores NaN detectados")
        
        # Retornar solo columnas necesarias
        result = df[['Date', variable_name]].copy()
        
        logger.info(f"{variable_name} procesado: {len(result)} observaciones "
                   f"({result['Date'].min().strftime('%Y-%m')} a {result['Date'].max().strftime('%Y-%m')})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error procesando {variable_name}: {e}")
        raise


def process_all_exogenous():
    """
    Procesa todas las variables exógenas desde raw/ a clean/
    """
    # Configuración de archivos
    files_config = {
        'USD_Index': 'Datos históricos Futuros Índice dólar.csv',
        'VIX': 'Datos históricos del S&P 500 VIX.csv',
        'Brent_Oil': 'Datos históricos Futuros petróleo Brent.csv',
        'Nickel': 'Datos históricos Futuros níquel.csv'
    }

    
    base_path = Path('data/exogenous')
    raw_path = base_path / 'raw'
    clean_path = base_path / 'clean'
    
    # Crear directorio clean si no existe
    clean_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== PROCESAMIENTO DE VARIABLES EXÓGENAS ===\n")
    
    results = {}
    
    for var_name, filename in files_config.items():
        filepath = raw_path / filename
        
        if not filepath.exists():
            logger.warning(f"{var_name}: Archivo no encontrado en {filepath}")
            continue
        
        try:
            # Procesar archivo
            df_clean = clean_investing_data(filepath, var_name)
            
            # Guardar archivo limpio
            output_file = clean_path / f"{var_name}.csv"
            df_clean.to_csv(output_file, index=False)
            
            # Guardar resumen
            results[var_name] = {
                'obs': len(df_clean),
                'start': df_clean['Date'].min(),
                'end': df_clean['Date'].max(),
                'nan_count': df_clean[var_name].isna().sum()
            }
            
        except Exception as e:
            logger.error(f"Error procesando {var_name}: {e}")
    
    # Reporte final
    logger.info("\n=== RESUMEN FINAL ===\n")
    for var, info in results.items():
        logger.info(f"{var}:")
        logger.info(f"  Observaciones: {info['obs']}")
        logger.info(f"  Rango: {info['start'].strftime('%Y-%m')} a {info['end'].strftime('%Y-%m')}")
        logger.info(f"  NaN: {info['nan_count']}\n")
    
    return results


if __name__ == '__main__':
    results = process_all_exogenous()
