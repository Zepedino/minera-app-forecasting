"""
Script principal para comparación de modelos de pronóstico
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importar módulos personalizados
from utils.data_loader import InvestingDataLoader
from utils.metrics import evaluate_forecast, print_metrics
from models.exponential_smoothing import ExponentialSmoothingModels
from models.arima_models import ARIMAModels

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class ForecastingComparison:
    """
    Clase principal para comparar modelos de pronóstico
    """
    
    def __init__(self, metal_name, train_data, test_data):
        self.metal_name = metal_name
        self.train = train_data
        self.test = test_data
        self.forecast_steps = len(test_data)
        self.results = []
        
    def run_all_models(self):
        """
        Ejecuta los 6 modelos y almacena resultados
        """
        print(f"\n{'='*60}")
        print(f"🔬 EVALUANDO MODELOS PARA {self.metal_name.upper()}")
        print(f"{'='*60}")
        
        # 1. Holt
        self._run_model("Holt", 
                       ExponentialSmoothingModels.fit_holt)
        
        # 2. Brown
        self._run_model("Brown", 
                       ExponentialSmoothingModels.fit_brown)
        
        # 3. Damped Trend
        self._run_model("Damped Trend", 
                       ExponentialSmoothingModels.fit_damped_trend)
        
        # 4. ARIMA
        self._run_model("ARIMA", 
                       lambda t, s: ARIMAModels.fit_auto_arima(t, s, seasonal=False))
        
        # 5. SARIMA
        self._run_model("SARIMA", 
                       lambda t, s: ARIMAModels.fit_auto_arima(t, s, seasonal=True, m=12))
        
        # 6. ARIMA Manual (1,1,1) - baseline
        self._run_model("ARIMA(1,1,1)", 
                       lambda t, s: ARIMAModels.fit_manual_arima(t, s, order=(1,1,1)))
        
        return self.results
    
    def _run_model(self, model_name, fit_function):
        """
        Ejecuta un modelo individual y almacena métricas
        """
        print(f"\n⚙️  Entrenando {model_name}...")
        
        try:
            forecast, fitted_model = fit_function(self.train, self.forecast_steps)
            
            if forecast is None:
                print(f"   ⚠️  {model_name} falló, saltando...")
                return
            
            # Calcular métricas
            metrics = evaluate_forecast(
                self.test.values, 
                forecast.values, 
                model_name=model_name
            )
            
            # Agregar información adicional
            metrics['Metal'] = self.metal_name
            metrics['Forecast'] = forecast
            metrics['Model_Object'] = fitted_model
            
            self.results.append(metrics)
            
            print_metrics(metrics)
            
        except Exception as e:
            print(f"   ❌ Error ejecutando {model_name}: {str(e)}")
    
    def get_best_model(self):
        """
        Identifica el mejor modelo según MAPE
        """
        if not self.results:
            return None
        
        best = min(self.results, key=lambda x: x['MAPE'])
        
        print(f"\n🏆 MEJOR MODELO: {best['Modelo']}")
        print(f"   MAPE: {best['MAPE']:.2f}%")
        
        return best
    
    def plot_comparison(self, save_path=None):
        """
        Genera gráfico comparativo de todos los modelos
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Subplot 1: Series temporales
        ax1 = axes[0]
        ax1.plot(self.train.index, self.train.values, 
                label='Entrenamiento', color='blue', linewidth=2)
        ax1.plot(self.test.index, self.test.values, 
                label='Test (Real)', color='black', linewidth=2, linestyle='--')
        
        colors = sns.color_palette("husl", len(self.results))
        
        for i, result in enumerate(self.results):
            forecast = result['Forecast']
            ax1.plot(self.test.index, forecast.values, 
                    label=f"{result['Modelo']} (MAPE: {result['MAPE']:.1f}%)",
                    color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax1.set_title(f'Comparación de Pronósticos - {self.metal_name.upper()}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Precio')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Métricas comparativas
        ax2 = axes[1]
        
        metrics_df = pd.DataFrame([{
            'Modelo': r['Modelo'],
            'MAPE': r['MAPE'],
            'MAE': r['MAE'],
            'RMSE': r['RMSE']
        } for r in self.results])
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax2.bar(x - width, metrics_df['MAPE'], width, label='MAPE (%)', alpha=0.8)
        ax2.bar(x, metrics_df['MAE'], width, label='MAE', alpha=0.8)
        ax2.bar(x + width, metrics_df['RMSE'], width, label='RMSE', alpha=0.8)
        
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('Valor de Métrica')
        ax2.set_title('Comparación de Métricas de Error', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado en: {save_path}")
        
        plt.show()

def main():
    print("🚀 SISTEMA DE EVALUACIÓN DE MODELOS DE PRONÓSTICO")
    print("   Proyecto: MineraApp - Investigación de Modelos")
    print("   Fuente: Investing.com (1990-2025)")
    print("="*70)
    
    # 1. Cargar datos desde CSVs de Investing.com
    loader = InvestingDataLoader(data_folder='data/raw')
    
    # Cargar todos los metales
    metals_data = loader.load_all_metals()
    
    if not metals_data:
        print("❌ No se pudieron cargar datos de metales")
        return
    
    # Mostrar resumen de datos cargados
    summary = loader.get_data_summary(metals_data)
    print("\n📋 RESUMEN DE DATOS CARGADOS:")
    print(summary.to_string(index=False))
    print()
    
    # 2. Almacenar resultados globales
    all_results = []
    
    # 3. Procesar cada metal
    for metal_name, data in metals_data.items():
        
        # División temporal adaptativa
        if metal_name == 'cobalto':
            # Cobalto tiene menos historia (desde 2010)
            train, test = loader.train_test_split(data, test_years=5, metal_name=metal_name)
        else:
            # Cobre/Oro/Plata: validar con últimos 7 años (2019-2025)
            train, test = loader.train_test_split(data, test_years=7, metal_name=metal_name)
        
        # Crear objeto de comparación
        comparison = ForecastingComparison(metal_name, train, test)
        
        # Ejecutar todos los modelos
        results = comparison.run_all_models()
        
        # Obtener mejor modelo
        best = comparison.get_best_model()
        
        # Generar gráfico
        save_path = f"results/{metal_name}_comparison.png"
        Path("results").mkdir(exist_ok=True)
        comparison.plot_comparison(save_path=save_path)
        
        # Agregar a resultados globales
        all_results.extend(results)
    
    # 4. Generar resumen consolidado
    summary_df = pd.DataFrame([{
        'Metal': r['Metal'],
        'Modelo': r['Modelo'],
        'MAPE': r['MAPE'],
        'MAE': r['MAE'],
        'RMSE': r['RMSE'],
        'SMAPE': r['SMAPE']
    } for r in all_results])
    
    # Guardar en Excel
    excel_path = 'results/model_comparison_summary.xlsx'
    summary_df.to_excel(excel_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ EVALUACIÓN COMPLETADA")
    print(f"📄 Resumen guardado en: {excel_path}")
    print(f"{'='*70}")
    
    # Mostrar tabla resumen
    print("\n📊 TABLA RESUMEN:")
    print(summary_df.to_string(index=False))
    
    # Mejores modelos por metal
    print("\n🏆 MEJORES MODELOS POR METAL:")
    best_by_metal = summary_df.loc[summary_df.groupby('Metal')['MAPE'].idxmin()]
    print(best_by_metal[['Metal', 'Modelo', 'MAPE']].to_string(index=False))

if __name__ == "__main__":
    main()
