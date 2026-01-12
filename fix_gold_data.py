"""
Corrige datos de oro combinando:
- Investing.com 1990-2009 (datos correctos)
- Yahoo Finance 2010-2026 (datos limpios)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime

print("🔧 CORRIGIENDO DATOS DE ORO")
print("="*70)

# PASO 1: Cargar CSV de Investing.com
print("\n📂 Cargando Investing.com...")
df_investing = pd.read_csv('data/raw/Gold-Futures-Historical-Data.csv')

# Limpiar formato
df_investing['Date'] = pd.to_datetime(df_investing['Date'], format='%m/%d/%Y')

if df_investing['Price'].dtype == 'object':
    df_investing['Price'] = df_investing['Price'].str.replace(',', '').astype(float)

df_investing = df_investing.sort_values('Date')
df_investing.set_index('Date', inplace=True)

# Filtrar solo 1990-2009 (datos buenos)
df_old = df_investing[df_investing.index < '2010-01-01'].copy()

print(f"   ✅ Datos 1990-2009: {len(df_old)} registros")
print(f"   📅 Rango: {df_old.index.min().strftime('%Y-%m-%d')} → {df_old.index.max().strftime('%Y-%m-%d')}")
print(f"   💰 Precio: ${df_old['Price'].min():.2f} - ${df_old['Price'].max():.2f}")

# PASO 2: Descargar de Yahoo Finance (2010-2026)
print("\n📥 Descargando Yahoo Finance (2010-2026)...")

try:
    # GC=F es el ticker de Gold Futures en Yahoo
    gold_yahoo = yf.download(
        'GC=F', 
        start='2010-01-01', 
        end='2026-01-31', 
        interval='1mo',
        progress=False
    )
    
    # Extraer solo la columna Close
    gold_yahoo = gold_yahoo[['Open', 'High', 'Low', 'Close']].copy()
    gold_yahoo.columns = ['Open', 'High', 'Low', 'Price']
    
    # Redondear a 2 decimales
    gold_yahoo = gold_yahoo.round(2)
    
    print(f"   ✅ Datos Yahoo: {len(gold_yahoo)} registros")
    print(f"   📅 Rango: {gold_yahoo.index.min().strftime('%Y-%m-%d')} → {gold_yahoo.index.max().strftime('%Y-%m-%d')}")
    print(f"   💰 Precio: ${gold_yahoo['Price'].min():.2f} - ${gold_yahoo['Price'].max():.2f}")
    
except Exception as e:
    print(f"   ❌ Error descargando Yahoo: {e}")
    exit(1)

# PASO 3: Combinar ambos DataFrames
print("\n🔗 Combinando datos...")

# Asegurar que Yahoo tiene las mismas columnas que Investing
for col in df_old.columns:
    if col not in gold_yahoo.columns:
        if col in ['Vol.', 'Change %']:
            gold_yahoo[col] = ''
        else:
            gold_yahoo[col] = gold_yahoo['Price']  # Llenar con Price si no existe

# Reordenar columnas
gold_yahoo = gold_yahoo[df_old.columns]

# Concatenar
df_combined = pd.concat([df_old, gold_yahoo])
df_combined = df_combined.sort_index()

# Eliminar duplicados (por si hay overlap)
df_combined = df_combined[~df_combined.index.duplicated(keep='last')]

print(f"   ✅ Total final: {len(df_combined)} registros")
print(f"   📅 Rango completo: {df_combined.index.min().strftime('%Y-%m-%d')} → {df_combined.index.max().strftime('%Y-%m-%d')}")
print(f"   💰 Precio: ${df_combined['Price'].min():.2f} - ${df_combined['Price'].max():.2f}")

# PASO 4: Verificar que no hay valores anormales
low_values = df_combined[df_combined['Price'] < 100]
if len(low_values) > 0:
    print(f"\n⚠️  ADVERTENCIA: {len(low_values)} valores < $100 detectados")
    print(low_values)
else:
    print(f"\n✅ Verificación OK: No hay valores anormales")

# PASO 5: Guardar CSV limpio
df_combined.reset_index(inplace=True)
df_combined['Date'] = df_combined['Date'].dt.strftime('%m/%d/%Y')

output_path = 'data/raw/Gold-Futures-Historical-Data-CLEAN.csv'
df_combined.to_csv(output_path, index=False)

print(f"\n💾 Archivo guardado: {output_path}")
print("\n" + "="*70)
print("✅ CORRECCIÓN COMPLETADA")
print("\nAhora ejecuta:")
print("   1. Renombra el archivo original como backup:")
print("      mv Gold-Futures-Historical-Data.csv Gold-Futures-Historical-Data-OLD.csv")
print("\n   2. Renombra el limpio:")
print("      mv Gold-Futures-Historical-Data-CLEAN.csv Gold-Futures-Historical-Data.csv")
print("\n   3. Re-ejecuta tu análisis:")
print("      python main.py")
print("="*70)

# PASO 6: Mostrar muestra del periodo crítico
print("\n📊 MUESTRA DEL PERIODO CORREGIDO (2017-2020):")
mask = (df_combined['Date'] >= '04/01/2017') & (df_combined['Date'] <= '01/01/2020')
sample = df_combined[mask][['Date', 'Price']].head(10)
print(sample.to_string(index=False))
