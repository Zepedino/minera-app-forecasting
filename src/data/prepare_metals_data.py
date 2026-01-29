"""
Procesa datos RAW de metales desde investing.com
Genera archivos individuales por metal en español
"""

import pandas as pd
from pathlib import Path
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapeo inglés -> español
METAL_NAMES = {
    'copper': 'cobre',
    'gold': 'oro',
    'silver': 'plata',
    'cobalt': 'cobalto'
}

def main():
    current = Path(__file__).resolve()
    dataraw = current.parent.parent.parent / "data" / "raw"
    
    logger.info(f"Buscando en: {dataraw}")
    
    if not dataraw.exists():
        logger.error(f"Carpeta no existe: {dataraw}")
        return
    
    files = sorted(dataraw.glob("*Futures-Historical-Data.csv"))
    logger.info(f"CSV encontrados: {[f.name for f in files]}")
    
    if not files:
        logger.error("No hay CSV en data/raw")
        return
    
    dfs_dict = {}
    
    for path in files:
        metal_en = path.stem.replace("-Futures-Historical-Data", "").lower()
        metal_es = METAL_NAMES.get(metal_en, metal_en)
        
        logger.info(f"\n=== Procesando {metal_es.upper()} ({metal_en}) ===")
        
        try:
            # LECTURA ESPECIAL PARA PLATA
            if metal_en == "silver":
                logger.info(f"{metal_es}: Lectura especial (CSV corrupto)")
                with open(path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
                    # Limpiar quotes rotos y delimitadores mal formados
                    content = content.replace('Date,Price"', 'Date,Price')
                    content = content.replace('Change %"', 'Change %')
                    content = content.replace(',"', ',')
                    content = content.replace('"', '')
                df = pd.read_csv(io.StringIO(content))
            else:
                # Lectura normal con encoding UTF-8-SIG para BOM
                df = pd.read_csv(path, encoding='utf-8-sig')
            
            logger.info(f"{metal_es} columnas RAW: {df.columns.tolist()}")
            
            # LIMPIEZA DE NOMBRES DE COLUMNAS
            df.columns = (df.columns
                .str.strip()
                .str.replace('"', '', regex=False)
                .str.replace(',', '', regex=False))
            
            logger.info(f"{metal_es} columnas LIMPIAS: {df.columns.tolist()}")
            
            # BÚSQUEDA FLEXIBLE DE COLUMNAS (case-insensitive)
            date_col = None
            price_col = None
            
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                if 'price' in col.lower():
                    price_col = col
            
            if date_col is None:
                logger.error(f"{metal_es}: No se encontró columna de fecha")
                logger.error(f" Columnas disponibles: {df.columns.tolist()}")
                continue
            
            if price_col is None:
                logger.error(f"{metal_es}: No se encontró columna de precio")
                logger.error(f" Columnas disponibles: {df.columns.tolist()}")
                continue
            
            logger.info(f"{metal_es}: usando Date='{date_col}', Price='{price_col}'")
            
            # LIMPIEZA DE FECHAS (formato mixto DD/MM/YYYY o MM/DD/YYYY)
            df[date_col] = pd.to_datetime(
                df[date_col],
                format='mixed',
                dayfirst=True,
                errors='coerce'
            )
            
            # LIMPIEZA DE PRECIOS
            df[price_col] = (df[price_col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('.', '', regex=False)
                .str.strip())
            
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            
            # RENAME Y FILTRO
            df = df.rename(columns={date_col: 'Date', price_col: 'Price'})
            df = df[['Date', 'Price']].dropna()
            df = df.sort_values('Date').reset_index(drop=True)
            
            # GUARDAR ARCHIVO INDIVIDUAL POR METAL EN ESPAÑOL
            output_dir = dataraw.parent / "metals"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{metal_es}_monthly.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(f"{metal_es}: {len(df)} filas, {df['Date'].min()} a {df['Date'].max()}")
            logger.info(f"Guardado en: {output_file}")
            
            dfs_dict[metal_es] = df
            
        except Exception as e:
            logger.error(f"{metal_es}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(dfs_dict) == 0:
        logger.error("No se pudieron procesar los CSV")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("RESUMEN FINAL")
    logger.info(f"{'='*60}")
    logger.info(f"Metales procesados: {list(dfs_dict.keys())}")
    logger.info(f"Archivos guardados en: data/metals/")
    
    for metal, df in dfs_dict.items():
        logger.info(f"  {metal}: {len(df)} observaciones")

if __name__ == "__main__":
    main()
