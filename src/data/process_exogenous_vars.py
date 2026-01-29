# src/data/process_exogenous_vars.py
"""
Procesa variables exógenas desde investing.com
Limpia formato europeo (comas, fechas DD/MM/YYYY) y exporta a CSV limpio
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_investing_csv(filepath, var_name):
    """
    Limpia CSV de investing.com y retorna DataFrame con columnas [Date, var_name]
    
    Args:
        filepath: Path al archivo CSV crudo
        var_name: Nombre de la variable (ej: 'USD_Index', 'VIX')
    
    Returns:
        DataFrame con columnas ['Date', var_name]
    """
    try:
        # Leer CSV con encoding para BOM
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        logger.info(f"{var_name} - Columnas RAW: {df.columns.tolist()}")
        
        # Limpiar nombres de columnas
        df.columns = (df.columns
                     .str.strip()
                     .str.replace('"', '', regex=False)
                     .str.replace('ú', 'u', regex=False)
                     .str.replace('ó', 'o', regex=False))
        
        # Buscar columnas de fecha y precio
        date_col = None
        price_col = None
        
        for col in df.columns:
            if 'fecha' in col.lower() or 'date' in col.lower():
                date_col = col
            if 'ultimo' in col.lower() or 'price' in col.lower() or 'último' in col.lower():
                price_col = col
        
        if date_col is None or price_col is None:
            logger.error(f"{var_name} - No se encontraron columnas necesarias")
            logger.error(f"  Columnas disponibles: {df.columns.tolist()}")
            return None
        
        logger.info(f"{var_name} - Usando Date='{date_col}', Price='{price_col}'")
        
        # Limpiar fechas (formato DD/MM/YYYY o DD.MM.YYYY de investing.com)
        df[date_col] = pd.to_datetime(
            df[date_col].astype(str).str.replace('.', '/', regex=False),
            format='mixed',
            dayfirst=True,
            errors='coerce'
        )
        
        # Limpiar precios (formato europeo: punto = miles, coma = decimal)
        # Ejemplo: "16.681,88" -> 16681.88
        df[price_col] = (df[price_col]
                        .astype(str)
                        .str.replace('.', '', regex=False)   # Quitar separador miles
                        .str.replace(',', '.', regex=False)  # Coma decimal -> punto
                        .str.strip())
        
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Rename y filtrar
        df = df.rename(columns={date_col: 'Date', price_col: var_name})
        df = df[['Date', var_name]].dropna()
        df = df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"{var_name} - {len(df)} filas, {df['Date'].min()} a {df['Date'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"{var_name} - ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Procesa todas las variables exógenas de investing.com"""
    
    base_path = Path(__file__).resolve().parent.parent.parent
    raw_path = base_path / "data" / "exogenous" / "raw"
    clean_path = base_path / "data" / "exogenous" / "clean"
    
    # Crear directorio clean si no existe
    clean_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Buscando archivos en: {raw_path}")
    
    # Configuración de archivos
    files_config = {
        'USD_Index': 'Datos históricos Futuros Índice dólar.csv',
        'VIX': 'Datos históricos del S&P 500 VIX.csv',
        'Brent_Oil': 'Datos históricos Futuros petróleo Brent.csv',
        'Nickel': 'Datos históricos Futuros níquel.csv'
    }
    
    results = {}
    
    for var_name, filename in files_config.items():
        filepath = raw_path / filename
        
        if not filepath.exists():
            logger.warning(f"{var_name} - Archivo no encontrado: {filename}")
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Procesando {var_name}")
        logger.info(f"{'='*50}")
        
        df_clean = clean_investing_csv(filepath, var_name)
        
        if df_clean is not None:
            # Guardar archivo limpio
            output_file = clean_path / f"{var_name}.csv"
            df_clean.to_csv(output_file, index=False)
            
            results[var_name] = {
                'obs': len(df_clean),
                'start': df_clean['Date'].min(),
                'end': df_clean['Date'].max(),
                'nan_count': df_clean[var_name].isna().sum()
            }
            
            logger.info(f"Guardado: {output_file}")
    
    # Reporte final
    logger.info(f"\n{'='*50}")
    logger.info("RESUMEN FINAL")
    logger.info(f"{'='*50}")
    
    for var, info in results.items():
        logger.info(f"\n{var}:")
        logger.info(f"  Observaciones: {info['obs']}")
        logger.info(f"  Rango: {info['start'].strftime('%Y-%m-%d')} a {info['end'].strftime('%Y-%m-%d')}")
        logger.info(f"  NaN detectados: {info['nan_count']}")
    
    logger.info(f"\nVariables limpias guardadas en: {clean_path}")


if __name__ == "__main__":
    main()
