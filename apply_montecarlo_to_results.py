"""
Script para aplicar Monte Carlo sobre resultados existentes
Lee el Excel generado por main.py y aplica simulación estocástica
"""
import pandas as pd
from pathlib import Path

# Importar módulos existentes
from utils.data_loader import InvestingDataLoader
from models.exponential_smoothing import ExponentialSmoothingModels
from models.arima_models import ARIMAModels
from models.monte_carlo_simulator import MonteCarloSimulator

# Diccionario de selección de modelo por metal (basado en tus resultados)
MODEL_SELECTION = {
    'cobalto': 'Damped Trend',  # MAPE 25.65% (mejor que Holt 26.94%)
    'cobre': 'Holt',            # MAPE 24.43% (mejor que Damped 27.79%)
    'oro': 'Holt',              # MAPE 24.76% (mejor que Damped 28.79%)
    'plata': 'Holt'             # MAPE 28.69% (mejor que Damped 32.20%)
}

def train_selected_model(model_name, train, forecast_steps):
    """
    Entrena el modelo seleccionado
    """
    if model_name == 'Damped Trend':
        return ExponentialSmoothingModels.fit_damped_trend(train, forecast_steps)
    elif model_name == 'Holt':
        return ExponentialSmoothingModels.fit_holt(train, forecast_steps)
    elif model_name == 'Brown':
        return ExponentialSmoothingModels.fit_brown(train, forecast_steps)
    elif model_name == 'ARIMA':
        return ARIMAModels.fit_auto_arima(train, forecast_steps, seasonal=False)
    elif model_name == 'SARIMA':
        return ARIMAModels.fit_auto_arima(train, forecast_steps, seasonal=True, m=12)
    elif model_name == 'ARIMA(1,1,1)':
        return ARIMAModels.fit_manual_arima(train, forecast_steps, order=(1,1,1))
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

def main():
    print("="*80)
    print("🎲 ANÁLISIS MONTE CARLO SOBRE RESULTADOS EXISTENTES")
    print("   Proyecto: MineraApp - Análisis de Incertidumbre")
    print("="*80)
    
    # PASO 1: Verificar que existan resultados previos
    results_file = Path('results/model_comparison_summary.xlsx')
    
    if not results_file.exists():
        print("\n❌ ERROR: No se encontró 'results/model_comparison_summary.xlsx'")
        print("   Ejecuta primero 'python main.py' para generar resultados base")
        return
    
    # PASO 2: Cargar resultados previos
    print("\n📂 Cargando resultados previos...")
    resultados_previos = pd.read_excel(results_file)
    print(f"   ✅ {len(resultados_previos)} resultados cargados")
    
    # PASO 3: Cargar datos originales
    loader = InvestingDataLoader(data_folder='data/raw')
    metals_data = loader.load_all_metals()
    
    # PASO 4: Crear carpeta para resultados Monte Carlo
    mc_folder = Path('results/montecarlo')
    mc_folder.mkdir(exist_ok=True)
    
    # PASO 5: Procesar cada metal
    all_mc_summary = []
    
    for metal_name, data in metals_data.items():
        
        print(f"\n{'='*80}")
        print(f"🔬 PROCESANDO: {metal_name.upper()}")
        print(f"{'='*80}")
        
        # Filtrar resultados de este metal
        metal_results = resultados_previos[resultados_previos['Metal'] == metal_name]
        
        if len(metal_results) == 0:
            print(f"   ⚠️  No hay resultados para {metal_name}, saltando...")
            continue
        
        # Seleccionar modelo según diccionario
        selected_model_name = MODEL_SELECTION.get(metal_name, 'Holt')
        
        print(f"\n   📊 Modelo seleccionado: {selected_model_name}")
        
        # Buscar MAPE del modelo seleccionado
        selected_row = metal_results[metal_results['Modelo'] == selected_model_name]
        
        if len(selected_row) == 0:
            print(f"   ⚠️  Modelo {selected_model_name} no encontrado, usando Holt")
            selected_model_name = 'Holt'
            selected_row = metal_results[metal_results['Modelo'] == 'Holt']
        
        selected_row = selected_row.iloc[0]
        print(f"   📈 MAPE: {selected_row['MAPE']:.2f}%")
        
        # Justificar selección
        if metal_name in ['cobre', 'oro', 'plata']:
            holt_row = metal_results[metal_results['Modelo'] == 'Holt'].iloc[0]
            damped_row = metal_results[metal_results['Modelo'] == 'Damped Trend'].iloc[0]
            diff = damped_row['MAPE'] - holt_row['MAPE']
            print(f"   💡 Razón: Holt supera a Damped por {diff:.2f} puntos de MAPE")
        else:
            print(f"   💡 Razón: Damped Trend es superior empíricamente")
        
        # División train-test (misma que en main.py)
        if metal_name == 'cobalto':
            train, test = loader.train_test_split(data, test_years=5, metal_name=metal_name)
        else:
            train, test = loader.train_test_split(data, test_years=7, metal_name=metal_name)
        
        # Re-entrenar solo el modelo seleccionado (rápido, 1 modelo no 6)
        print(f"\n   ⚙️  Re-entrenando {selected_model_name}...")
        forecast, fitted_model = train_selected_model(selected_model_name, train, len(test))
        
        if forecast is None:
            print(f"   ❌ Error entrenando {selected_model_name}, saltando...")
            continue
        
        print(f"   ✅ Modelo entrenado exitosamente")
        
        # PASO 6: Aplicar Monte Carlo
        mc_simulator = MonteCarloSimulator(
            train_data=train,
            test_data=test,
            base_forecast=forecast,
            model_name=selected_model_name,
            mape_baseline=selected_row['MAPE'],
            n_simulations=10000
        )
        
        # Ejecutar simulaciones
        mc_results = mc_simulator.run_simulations()
        
        # Generar gráfico Fan Chart
        mc_simulator.plot_fan_chart(
            metal_name=metal_name,
            save_path=f"results/montecarlo/{metal_name}_montecarlo_analysis.png"
        )
        
        # Generar reporte de texto
        report = mc_simulator.generate_report(
            metal_name=metal_name,
            save_path=f"results/montecarlo/{metal_name}_probability_report.txt"
        )
        
        # Guardar bandas de confianza en CSV
        bands_df = pd.DataFrame({
            'Fecha': test.index,
            'Real': test.values,
            'Forecast': forecast.values,
            'P5': mc_results['percentile_5'].values,
            'P25': mc_results['percentile_25'].values,
            'P50': mc_results['percentile_50'].values,
            'P75': mc_results['percentile_75'].values,
            'P95': mc_results['percentile_95'].values
        })
        bands_df.to_csv(f"results/montecarlo/{metal_name}_confidence_bands.csv", index=False)
        
        # Resumen para consolidado
        final_prices = mc_results['simulations'].iloc[:, -1]
        all_mc_summary.append({
            'Metal': metal_name,
            'Modelo': selected_model_name,
            'MAPE': selected_row['MAPE'],
            'Precio_Forecast': forecast.iloc[-1],
            'Precio_Real': test.iloc[-1],
            'MC_Mean': final_prices.mean(),
            'MC_Median': final_prices.median(),
            'MC_Std': final_prices.std(),
            'MC_P5': final_prices.quantile(0.05),
            'MC_P95': final_prices.quantile(0.95),
            'IC90_Range': final_prices.quantile(0.95) - final_prices.quantile(0.05)
        })
        
        print(f"\n   ✅ Análisis Monte Carlo completado para {metal_name}")
    
    # PASO 7: Generar resumen consolidado
    if all_mc_summary:
        summary_df = pd.DataFrame(all_mc_summary)
        summary_df.to_excel('results/montecarlo/montecarlo_summary.xlsx', index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ ANÁLISIS MONTE CARLO COMPLETADO")
        print(f"{'='*80}")
        print(f"\n📊 RESUMEN:")
        print(summary_df.to_string(index=False))
        print(f"\n📁 Archivos generados:")
        print(f"   - Gráficos Fan Chart: results/montecarlo/*_montecarlo_analysis.png")
        print(f"   - Reportes texto: results/montecarlo/*_probability_report.txt")
        print(f"   - Bandas confianza: results/montecarlo/*_confidence_bands.csv")
        print(f"   - Resumen Excel: results/montecarlo/montecarlo_summary.xlsx")
        print(f"\n{'='*80}")
    else:
        print("\n❌ No se generaron resultados Monte Carlo")

if __name__ == "__main__":
    main()
