"""
Script para evaluar modelos Prophet y ARIMAX
Complementa main.py agregando modelos avanzados con detección de cambios estructurales
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar módulos existentes
from utils.data_loader import InvestingDataLoader
from utils.metrics import evaluate_forecast, print_metrics

# Importar nuevos modelos
from models.prophet_model import ProphetModels
from models.arimax_models import ARIMAXModels

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class AdvancedForecastingEvaluation:
    """
    Evaluación de Prophet y ARIMAX sobre datos de commodities
    """
    
    def __init__(self, metal_name, train_data, test_data):
        self.metal_name = metal_name
        self.train = train_data
        self.test = test_data
        self.forecast_steps = len(test_data)
        self.results = []
        
    def _prepare_exogenous_variables(self):
        """
        Prepara variables exógenas para ARIMAX
        
        Variables simuladas (en producción usar APIs reales):
        - PMI: Purchasing Managers Index (demanda global)
        - USD: USD Index (efecto moneda)
        - Kilian: Real Activity Index
        
        Referencias:
        - Balioz et al. (2024): Variables exógenas para commodities
        """
        n_periods = len(self.train)
        
        # PMI: Global Manufacturing PMI (mean=50, std=5)
        # Valores realistas: recesión=40-45, expansión=55-60
        pmi = np.random.normal(50, 5, n_periods)
        pmi = np.clip(pmi, 30, 70)
        
        # USD Index: base 100, volatilidad mensual ~0.3%
        usd = 100 + np.cumsum(np.random.normal(0, 0.3, n_periods))
        
        # Kilian Real Activity Index: normalizado, mean=0, std=1
        kilian = np.random.normal(0, 1, n_periods)
        
        exog_df = pd.DataFrame({
            'PMI': pmi,
            'USD_Index': usd,
            'Kilian_RAI': kilian
        }, index=self.train.index)
        
        return exog_df
    
    def evaluate_prophet(self):
        """
        Evalúa Prophet con detección automática de changepoints
        """
        print(f"\n⚙️  Entrenando Prophet...")
        
        try:
            forecast, model = ProphetModels.fit_prophet_base(
                self.train, 
                self.forecast_steps
            )
            
            if forecast is None:
                print(f"   ⚠️  Prophet falló, saltando...")
                return None
            
            # Calcular métricas
            metrics = evaluate_forecast(
                self.test.values,
                forecast.values,
                model_name="Prophet"
            )
            
            # Detectar changepoints
            changepoints = ProphetModels.get_changepoint_dates(model)
            
            # Obtener intervalos de confianza
            _, lower, upper = ProphetModels.get_forecast_intervals(
                model, 
                self.forecast_steps
            )
            
            metrics['Metal'] = self.metal_name
            metrics['Forecast'] = forecast
            metrics['Changepoints_Count'] = len(changepoints)
            metrics['Changepoints_Dates'] = changepoints.tolist()
            metrics['CI_Lower'] = lower
            metrics['CI_Upper'] = upper
            
            self.results.append(metrics)
            
            print_metrics(metrics)
            print(f"   📍 Changepoints detectados: {len(changepoints)}")
            
            if len(changepoints) > 0:
                recent_changes = changepoints[-3:]
                print(f"   📅 Últimos cambios: {[str(d.date()) for d in recent_changes]}")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error en Prophet: {str(e)}")
            return None
    
    def evaluate_arimax(self):
        """
        Evalúa ARIMAX con variables exógenas
        """
        print(f"\n⚙️  Entrenando ARIMAX...")
        
        try:
            # Preparar variables exógenas
            exog = self._prepare_exogenous_variables()
            
            print(f"   📊 Variables exógenas: {list(exog.columns)}")
            
            # Entrenar ARIMAX
            forecast, model = ARIMAXModels.fit_arimax_auto(
                self.train,
                exog,
                self.forecast_steps,
                seasonal=True,
                m=12
            )
            
            if forecast is None:
                print(f"   ⚠️  ARIMAX falló, saltando...")
                return None
            
            # Calcular métricas
            metrics = evaluate_forecast(
                self.test.values,
                forecast.values,
                model_name="ARIMAX"
            )
            
            metrics['Metal'] = self.metal_name
            metrics['Forecast'] = forecast
            metrics['Exogenous_Variables'] = list(exog.columns)
            
            # Obtener parámetros del modelo
            try:
                aic = model.aic
                bic = model.bic
                metrics['AIC'] = aic
                metrics['BIC'] = bic
                print(f"   📐 AIC: {aic:.2f} | BIC: {bic:.2f}")
            except:
                pass
            
            self.results.append(metrics)
            print_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error en ARIMAX: {str(e)}")
            return None
    
    def evaluate_prophet_with_holidays(self):
        """
        Evalúa Prophet con eventos históricos específicos
        """
        print(f"\n⚙️  Entrenando Prophet + Holidays...")
        
        try:
            # Definir eventos históricos conocidos
            holidays = pd.DataFrame({
                'holiday': ['COVID_crash', 'Ukraine_war', 'China_lockdown'],
                'ds': pd.to_datetime(['2020-03-15', '2022-02-24', '2022-09-01']),
                'lower_window': [-30, -10, -15],
                'upper_window': [90, 120, 60]
            })
            
            forecast, model = ProphetModels.fit_prophet_with_holidays(
                self.train,
                self.forecast_steps,
                holidays_dict=holidays
            )
            
            if forecast is None:
                print(f"   ⚠️  Prophet+Holidays falló, saltando...")
                return None
            
            # Calcular métricas
            metrics = evaluate_forecast(
                self.test.values,
                forecast.values,
                model_name="Prophet+Holidays"
            )
            
            metrics['Metal'] = self.metal_name
            metrics['Forecast'] = forecast
            metrics['Holidays_Count'] = len(holidays)
            
            self.results.append(metrics)
            print_metrics(metrics)
            print(f"   🎯 Eventos capturados: COVID-19, Ukraine War, China Lockdown")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error en Prophet+Holidays: {str(e)}")
            return None
    
    def compare_arimax_vs_arima(self):
        """
        Compara ARIMAX (con exógenas) vs ARIMA (univariado)
        Para demostrar mejora empírica
        """
        print(f"\n🔬 Comparando ARIMAX vs ARIMA univariado...")
        
        try:
            exog = self._prepare_exogenous_variables()
            
            comparison = ARIMAXModels.compare_arimax_vs_arima(
                self.train,
                exog,
                self.forecast_steps
            )
            
            print(f"\n   📊 RESULTADOS COMPARACIÓN:")
            print(f"   ├─ ARIMA (univariado):  MAPE = {comparison['arima_mape']:.2f}%")
            print(f"   ├─ ARIMAX (con exógenas): MAPE = {comparison['arimax_mape']:.2f}%")
            print(f"   └─ Mejora: {comparison['improvement_percent']:.1f}%")
            
            return comparison
            
        except Exception as e:
            print(f"   ❌ Error en comparación: {str(e)}")
            return None
    
    def plot_advanced_comparison(self, save_path=None):
        """
        Genera gráfico comparativo de modelos avanzados
        """
        if not self.results:
            print("⚠️  No hay resultados para graficar")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Subplot 1: Series temporales con intervalos de confianza
        ax1 = axes[0]
        ax1.plot(self.train.index, self.train.values,
                label='Entrenamiento', color='blue', linewidth=2, alpha=0.7)
        ax1.plot(self.test.index, self.test.values,
                label='Test (Real)', color='black', linewidth=2.5, linestyle='--')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, result in enumerate(self.results):
            forecast = result['Forecast']
            model_name = result['Modelo']
            mape = result['MAPE']
            
            ax1.plot(self.test.index, forecast.values,
                    label=f"{model_name} (MAPE: {mape:.1f}%)",
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.8)
            
            # Si tiene intervalos de confianza (Prophet)
            if 'CI_Lower' in result and 'CI_Upper' in result:
                ax1.fill_between(
                    self.test.index,
                    result['CI_Lower'],
                    result['CI_Upper'],
                    color=colors[i % len(colors)],
                    alpha=0.15,
                    label=f"{model_name} IC 90%"
                )
        
        ax1.set_title(f'Modelos Avanzados: Prophet + ARIMAX - {self.metal_name.upper()}',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Precio')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Comparación de métricas
        ax2 = axes[1]
        
        metrics_df = pd.DataFrame([{
            'Modelo': r['Modelo'],
            'MAPE': r['MAPE'],
            'RMSE': r['RMSE'],
            'MAE': r['MAE']
        } for r in self.results])
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax2.bar(x - width, metrics_df['MAPE'], width,
               label='MAPE (%)', color='#FF6B6B', alpha=0.8)
        ax2.bar(x, metrics_df['MAE'], width,
               label='MAE', color='#4ECDC4', alpha=0.8)
        ax2.bar(x + width, metrics_df['RMSE'], width,
               label='RMSE', color='#45B7D1', alpha=0.8)
        
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('Valor de Métrica')
        ax2.set_title('Comparación de Métricas - Modelos Avanzados',
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_df['Modelo'], rotation=15, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Gráfico guardado en: {save_path}")
        
        plt.close()

def main():
    print("="*80)
    print("🚀 EVALUACIÓN DE MODELOS AVANZADOS: PROPHET + ARIMAX")
    print("   Proyecto: MineraApp - Detección de Cambios Estructurales")
    print("   Modelos: Prophet (changepoints) + ARIMAX (variables exógenas)")
    print("="*80)
    
    # 1. Cargar datos
    loader = InvestingDataLoader(data_folder='data/raw')
    metals_data = loader.load_all_metals()
    
    if not metals_data:
        print("❌ No se pudieron cargar datos")
        return
    
    # 2. Crear carpeta de resultados
    results_folder = Path('results/advanced_models')
    results_folder.mkdir(exist_ok=True, parents=True)
    
    # 3. Almacenar resultados globales
    all_results = []
    
    # 4. Procesar cada metal
    for metal_name, data in metals_data.items():
        
        print(f"\n{'='*80}")
        print(f"🔬 EVALUANDO: {metal_name.upper()}")
        print(f"{'='*80}")
        
        # División train-test
        if metal_name == 'cobalto':
            train, test = loader.train_test_split(data, test_years=5, metal_name=metal_name)
        else:
            train, test = loader.train_test_split(data, test_years=7, metal_name=metal_name)
        
        print(f"   📊 Train: {len(train)} obs | Test: {len(test)} obs")
        
        # Crear evaluador
        evaluator = AdvancedForecastingEvaluation(metal_name, train, test)
        
        # Evaluar modelos
        prophet_result = evaluator.evaluate_prophet()
        arimax_result = evaluator.evaluate_arimax()
        prophet_holidays_result = evaluator.evaluate_prophet_with_holidays()
        
        # Comparación ARIMAX vs ARIMA
        comparison = evaluator.compare_arimax_vs_arima()
        
        # Generar gráfico
        save_path = results_folder / f"{metal_name}_advanced_comparison.png"
        evaluator.plot_advanced_comparison(save_path=save_path)
        
        # Agregar a resultados globales
        all_results.extend(evaluator.results)
    
    # 5. Generar resumen consolidado
    if all_results:
        summary_df = pd.DataFrame([{
            'Metal': r['Metal'],
            'Modelo': r['Modelo'],
            'MAPE': r['MAPE'],
            'MAE': r['MAE'],
            'RMSE': r['RMSE'],
            'SMAPE': r['SMAPE']
        } for r in all_results])
        
        # Guardar en Excel
        excel_path = results_folder / 'advanced_models_summary.xlsx'
        summary_df.to_excel(excel_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ EVALUACIÓN COMPLETADA")
        print(f"{'='*80}")
        print(f"\n📊 RESUMEN:")
        print(summary_df.to_string(index=False))
        
        # Mejores modelos por metal
        print(f"\n🏆 MEJORES MODELOS AVANZADOS POR METAL:")
        best_by_metal = summary_df.loc[summary_df.groupby('Metal')['MAPE'].idxmin()]
        print(best_by_metal[['Metal', 'Modelo', 'MAPE']].to_string(index=False))
        
        print(f"\n📁 Archivos generados:")
        print(f"   - Resumen: {excel_path}")
        print(f"   - Gráficos: results/advanced_models/*_advanced_comparison.png")
        print(f"\n{'='*80}")
    else:
        print("\n❌ No se generaron resultados")

if __name__ == "__main__":
    main()
