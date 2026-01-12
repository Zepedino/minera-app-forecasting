"""
Módulo para carga de datos de metales desde CSVs de Investing.com
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class InvestingDataLoader:
    """
    Cargador de datos desde CSVs de Investing.com (1990-2025)
    """
    
    def __init__(self, data_folder='data/raw'):
        """
        Args:
            data_folder: Carpeta donde están los CSVs
        """
        self.data_folder = Path(data_folder)
        
    def load_csv(self, filepath):
        """
        Carga un CSV de Investing.com y lo procesa
        
        Args:
            filepath: Ruta al archivo CSV
            
        Returns:
            Serie temporal con precios mensuales
        """
        try:
            print(f"📂 Cargando {filepath.name}...")
            
            # Leer CSV (encoding para eliminar BOM: ﻿)
            df = pd.read_csv(
                filepath,
                encoding='utf-8-sig'
            )
            
            # Convertir Date a datetime (formato MM/DD/YYYY de Investing.com)
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            
            # Limpiar columna Price (remover comas de miles)
            if df['Price'].dtype == 'object':
                df['Price'] = df['Price'].str.replace(',', '').astype(float)
            
            # Establecer fecha como índice
            df = df.set_index('Date')
            
            # IMPORTANTE: Ordenar por fecha ASCENDENTE (Investing viene descendente)
            df = df.sort_index()
            
            # Tomar solo columna Price
            series = df['Price']
            
            # Remover NaNs si hay
            series = series.dropna()
            
            print(f"✅ {len(series)} observaciones mensuales")
            print(f"   📅 Rango: {series.index.min().strftime('%Y-%m')} → {series.index.max().strftime('%Y-%m')}")
            print(f"   ⏱️  Duración: {(series.index.max() - series.index.min()).days / 365.25:.1f} años")
            print(f"   💵 Min: ${series.min():.2f} | Max: ${series.max():.2f} | Actual: ${series.iloc[-1]:.2f}")
            
            return series
            
        except Exception as e:
            print(f"❌ Error cargando {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_metal(self, metal_name):
        """
        Carga datos de un metal específico
        
        Args:
            metal_name: 'cobre', 'oro', 'plata', 'cobalto'
            
        Returns:
            Serie temporal
        """
        metal_files = {
            'cobre': 'Copper-Futures-Historical-Data.csv',
            'oro': 'Gold-Futures-Historical-Data.csv',
            'plata': 'Silver-Futures-Historical-Data.csv',
            'cobalto': 'Cobalt-Futures-Historical-Data.csv'
        }
        
        filename = metal_files.get(metal_name.lower())
        
        if not filename:
            raise ValueError(f"Metal '{metal_name}' no soportado. Use: {list(metal_files.keys())}")
        
        filepath = self.data_folder / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        return self.load_csv(filepath)
    
    def load_all_metals(self):
        """
        Carga todos los metales disponibles
        
        Returns:
            dict: {metal: Serie temporal}
        """
        metals = ['cobre', 'oro', 'plata', 'cobalto']
        metals_data = {}
        
        print(f"\n{'='*70}")
        print("📥 CARGANDO DATOS HISTÓRICOS DE INVESTING.COM")
        print(f"{'='*70}\n")
        
        for metal in metals:
            try:
                data = self.load_metal(metal)
                if data is not None and len(data) > 0:
                    metals_data[metal] = data
                    print()
            except Exception as e:
                print(f"⚠️  No se pudo cargar {metal}: {str(e)}\n")
        
        print(f"{'='*70}")
        print(f"✅ {len(metals_data)} metales cargados exitosamente")
        print(f"{'='*70}\n")
        
        return metals_data
    
    @staticmethod
    def train_test_split(data, test_years=7, metal_name=''):
        """
        División temporal (respeta el orden cronológico)
        
        Args:
            data: Serie temporal
            test_years: Años para test (default 7 = 2019-2025)
            metal_name: Nombre del metal (para logging)
            
        Returns:
            train, test
        """
        # Calcular índice de corte basado en años
        test_months = test_years * 12
        split_idx = len(data) - test_months
        
        # Validar que quede suficiente data de entrenamiento
        min_train_months = 120  # Mínimo 10 años
        
        if split_idx < min_train_months:
            print(f"⚠️  {metal_name}: Ajustando split (data limitada)")
            # Usar 67-33 split para metales con menos historia
            split_idx = int(len(data) * 0.67)
        
        train = data[:split_idx]
        test = data[split_idx:]
        
        print(f"\n📊 División Train-Test para {metal_name.upper() if metal_name else 'Dataset'}:")
        print(f"   🔵 Train: {len(train)} obs | {train.index.min().strftime('%Y-%m')} → {train.index.max().strftime('%Y-%m')} ({len(train)/12:.1f} años)")
        print(f"   🟢 Test:  {len(test)} obs | {test.index.min().strftime('%Y-%m')} → {test.index.max().strftime('%Y-%m')} ({len(test)/12:.1f} años)")
        
        return train, test
    
    def get_data_summary(self, metals_data):
        """
        Genera resumen estadístico de los datos cargados
        
        Args:
            metals_data: dict con series temporales
            
        Returns:
            DataFrame con resumen
        """
        summary_data = []
        
        for metal, series in metals_data.items():
            años = (series.index.max() - series.index.min()).days / 365.25
            
            summary_data.append({
                'Metal': metal.capitalize(),
                'Observaciones': len(series),
                'Desde': series.index.min().strftime('%Y-%m'),
                'Hasta': series.index.max().strftime('%Y-%m'),
                'Años': f"{años:.1f}",
                'Precio_Min': f"{series.min():.2f}",
                'Precio_Max': f"{series.max():.2f}",
                'Precio_Actual': f"{series.iloc[-1]:.2f}",
                'Volatilidad_%': f"{(series.std() / series.mean() * 100):.1f}"
            })
        
        return pd.DataFrame(summary_data)

