"""
Módulo de Simulación Monte Carlo para Análisis de Incertidumbre
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class MonteCarloSimulator:
    """
    Simulador Monte Carlo para análisis probabilístico de pronósticos
    """
    
    def __init__(self, train_data, test_data, base_forecast, 
                 model_name, mape_baseline, n_simulations=10000):
        """
        Inicializa el simulador Monte Carlo
        
        Args:
            train_data: Serie de entrenamiento (para calcular volatilidad)
            test_data: Serie de test (para validación)
            base_forecast: Pronóstico base del modelo determinístico
            model_name: Nombre del modelo base (ej: "Holt", "Damped Trend")
            mape_baseline: MAPE del modelo determinístico
            n_simulations: Número de simulaciones (default 10,000)
        """
        self.train = train_data
        self.test = test_data
        self.forecast_base = base_forecast
        self.model_name = model_name
        self.mape_baseline = mape_baseline
        self.n_sims = n_simulations
        
        # Calcular estadísticas de la serie de entrenamiento
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()
        
        # Calcular volatilidad (desviación estándar de retornos porcentuales)
        returns = train_data.pct_change().dropna()
        self.volatility = returns.std()
        self.mean_return = returns.mean()
        
        # Almacenar resultados
        self.mc_results = None
        
        print(f"\n   📊 Parámetros Monte Carlo:")
        print(f"      - Volatilidad histórica: {self.volatility*100:.2f}% mensual")
        print(f"      - Retorno promedio: {self.mean_return*100:.2f}% mensual")
        print(f"      - Desv. estándar precios: ${self.train_std:.2f}")
    
    def run_simulations(self):
        """
        Ejecuta simulaciones Monte Carlo
        
        Returns:
            dict con percentiles, distribuciones y estadísticas
        """
        print(f"   🎲 Ejecutando {self.n_sims:,} simulaciones Monte Carlo...")
        
        simulations = []
        forecast_steps = len(self.forecast_base)
        
        np.random.seed(42)  # Reproducibilidad
        
        for sim in range(self.n_sims):
            simulated_path = []
            
            for t in range(forecast_steps):
                # Precio base del forecast determinístico
                base_price = self.forecast_base.iloc[t] if hasattr(self.forecast_base, 'iloc') else self.forecast_base[t]
                
                # Shock estocástico (distribución normal)
                shock = np.random.normal(self.mean_return, self.volatility)
                
                # Precio simulado (con límite inferior en 0)
                price_sim = max(0, base_price * (1 + shock))
                simulated_path.append(price_sim)
            
            simulations.append(simulated_path)
        
        # Convertir a DataFrame para análisis
        simulations_df = pd.DataFrame(simulations)
        
        # Calcular estadísticas por cada paso temporal
        results = {
            'simulations': simulations_df,
            'percentile_5': simulations_df.quantile(0.05, axis=0),
            'percentile_10': simulations_df.quantile(0.10, axis=0),
            'percentile_25': simulations_df.quantile(0.25, axis=0),
            'percentile_50': simulations_df.quantile(0.50, axis=0),  # Mediana
            'percentile_75': simulations_df.quantile(0.75, axis=0),
            'percentile_90': simulations_df.quantile(0.90, axis=0),
            'percentile_95': simulations_df.quantile(0.95, axis=0),
            'mean': simulations_df.mean(axis=0),
            'std': simulations_df.std(axis=0)
        }
        
        print(f"   ✅ Simulaciones completadas")
        
        self.mc_results = results
        return results
    
    def plot_fan_chart(self, metal_name='Metal', save_path=None):
        """
        Genera Fan Chart (gráfico de abanico) con intervalos de confianza
        """
        if self.mc_results is None:
            raise ValueError("Debe ejecutar run_simulations() primero")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # SUBPLOT 1: Fan Chart con bandas de confianza
        ax1 = axes[0]
        
        # Datos históricos (train)
        ax1.plot(self.train.index, self.train.values, 
                label='Entrenamiento', color='blue', linewidth=2, alpha=0.8)
        
        # Datos reales (test)
        ax1.plot(self.test.index, self.test.values, 
                label='Test (Real)', color='black', linewidth=2.5, linestyle='--')
        
        # Forecast base (determinístico)
        ax1.plot(self.test.index, self.forecast_base.values,
                label=f'{self.model_name} - Determinístico (MAPE: {self.mape_baseline:.1f}%)',
                color='red', linewidth=2.5, zorder=5)
        
        # Bandas de confianza Monte Carlo
        # IC 90% (P5-P95) - Más claro
        ax1.fill_between(
            self.test.index,
            self.mc_results['percentile_5'].values,
            self.mc_results['percentile_95'].values,
            alpha=0.15, color='orange', label='IC 90% (Monte Carlo)'
        )
        
        # IC 80% (P10-P90)
        ax1.fill_between(
            self.test.index,
            self.mc_results['percentile_10'].values,
            self.mc_results['percentile_90'].values,
            alpha=0.20, color='orange', label='IC 80% (Monte Carlo)'
        )
        
        # IC 50% (P25-P75) - Más oscuro
        ax1.fill_between(
            self.test.index,
            self.mc_results['percentile_25'].values,
            self.mc_results['percentile_75'].values,
            alpha=0.30, color='orange', label='IC 50% (Monte Carlo)'
        )
        
        # Mediana Monte Carlo
        ax1.plot(self.test.index, self.mc_results['percentile_50'].values,
                color='orange', linewidth=2, linestyle=':', label='Mediana MC', alpha=0.8)
        
        ax1.set_title(f'Pronóstico con Análisis de Incertidumbre - {metal_name.upper()}\n' + 
                     f'Modelo: {self.model_name} | Simulaciones: {self.n_sims:,}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Fecha', fontsize=11)
        ax1.set_ylabel('Precio', fontsize=11)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # SUBPLOT 2: Distribución de precio final
        ax2 = axes[1]
        
        # Obtener precios simulados del último mes
        final_prices = self.mc_results['simulations'].iloc[:, -1]
        
        # Histograma
        ax2.hist(final_prices, bins=60, alpha=0.7, color='steelblue', 
                edgecolor='black', density=True)
        
        # Líneas de percentiles clave
        percentiles = [
            (0.05, 'P5 (Pesimista)', 'red'),
            (0.25, 'P25', 'orange'),
            (0.50, 'P50 (Mediana)', 'green'),
            (0.75, 'P75', 'orange'),
            (0.95, 'P95 (Optimista)', 'red')
        ]
        
        for p, label, color in percentiles:
            value = final_prices.quantile(p)
            ax2.axvline(value, color=color, linestyle='--', linewidth=2, 
                       label=f'{label}: ${value:.0f}')
        
        # Precio real del último mes (si existe)
        if len(self.test) > 0:
            real_final = self.test.iloc[-1]
            ax2.axvline(real_final, color='black', linestyle='-', linewidth=2.5,
                       label=f'Real: ${real_final:.0f}')
        
        # Forecast determinístico
        forecast_final = self.forecast_base.iloc[-1] if hasattr(self.forecast_base, 'iloc') else self.forecast_base[-1]
        ax2.axvline(forecast_final, color='red', linestyle='-', linewidth=2.5,
                   label=f'Forecast: ${forecast_final:.0f}')
        
        ax2.set_title(f'Distribución Probabilística - Precio Final (Mes {len(self.forecast_base)})',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Precio', fontsize=11)
        ax2.set_ylabel('Densidad de Probabilidad', fontsize=11)
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Gráfico guardado: {save_path}")
        
        plt.close()
    
    def generate_report(self, metal_name='Metal', save_path=None):
        """
        Genera reporte probabilístico en formato texto
        """
        if self.mc_results is None:
            raise ValueError("Debe ejecutar run_simulations() primero")
        
        final_month = len(self.forecast_base)
        final_prices = self.mc_results['simulations'].iloc[:, -1]
        
        # Calcular probabilidades de escenarios
        forecast_final = self.forecast_base.iloc[-1] if hasattr(self.forecast_base, 'iloc') else self.forecast_base[-1]
        prob_above_forecast = (final_prices > forecast_final).mean() * 100
        prob_below_forecast = (final_prices < forecast_final).mean() * 100
        
        # Precio real (si existe)
        real_price_text = ""
        if len(self.test) > 0:
            real_final = self.test.iloc[-1]
            within_ic90 = (self.mc_results['percentile_5'].iloc[-1] <= real_final <= self.mc_results['percentile_95'].iloc[-1])
            real_price_text = f"""
VALIDACIÓN CON DATOS REALES:
  - Precio real (test):      ${real_final:.2f}
  - Dentro de IC90%:         {'SÍ ✅' if within_ic90 else 'NO ❌'}
  - Error vs Mediana MC:     ${abs(real_final - self.mc_results['percentile_50'].iloc[-1]):.2f}
"""
        
        report = f"""
{'='*75}
ANÁLISIS DE INCERTIDUMBRE MONTE CARLO - {metal_name.upper()}
{'='*75}

CONFIGURACIÓN:
  - Simulaciones:            {self.n_sims:,}
  - Modelo base:             {self.model_name}
  - MAPE baseline:           {self.mape_baseline:.2f}%
  - Volatilidad histórica:   {self.volatility*100:.2f}% mensual
  - Horizonte pronóstico:    {final_month} meses

{'='*75}
PRONÓSTICO MES {final_month} (ÚLTIMO MES):
{'='*75}

FORECAST DETERMINÍSTICO:
  - Precio Forecast:         ${forecast_final:.2f}

ANÁLISIS PROBABILÍSTICO (MONTE CARLO):
  - Precio Esperado (Media): ${final_prices.mean():.2f}
  - Mediana:                 ${final_prices.median():.2f}
  - Desviación Estándar:     ${final_prices.std():.2f}
  - Coef. Variación:         {(final_prices.std()/final_prices.mean())*100:.1f}%

INTERVALOS DE CONFIANZA:
  - P5  (Pesimista):         ${final_prices.quantile(0.05):.2f}
  - P10:                     ${final_prices.quantile(0.10):.2f}
  - P25:                     ${final_prices.quantile(0.25):.2f}
  - P50 (Mediana):           ${final_prices.quantile(0.50):.2f}
  - P75:                     ${final_prices.quantile(0.75):.2f}
  - P90:                     ${final_prices.quantile(0.90):.2f}
  - P95 (Optimista):         ${final_prices.quantile(0.95):.2f}

ANÁLISIS DE RIESGO:
  - Rango IC 90%:            ${final_prices.quantile(0.95) - final_prices.quantile(0.05):.2f}
  - Rango IC 50% (IQR):      ${final_prices.quantile(0.75) - final_prices.quantile(0.25):.2f}

PROBABILIDADES DE ESCENARIOS:
  - Prob(Precio > Forecast): {prob_above_forecast:.1f}%
  - Prob(Precio < Forecast): {prob_below_forecast:.1f}%
{real_price_text}
{'='*75}
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"   📄 Reporte guardado: {save_path}")
        
        return report
