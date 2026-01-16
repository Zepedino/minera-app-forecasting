"""
Configuraciones de modelos de forecasting.

Referencias academicas:
- Kahraman & Akay (2022): Exponential Smoothing para metales
- Taylor & Letham (2018): Prophet - Forecasting at Scale
- Cardozo et al. (2022): Monte Carlo en proyectos mineros
- Deveci (2013): ARIMAX para prediccion de oro y plata con variables macro
"""

# ============================================================================
# SUAVIZAMIENTO EXPONENCIAL
# ============================================================================

EXPONENTIAL_SMOOTHING = {
    'holt': {
        'initialization_method': 'estimated',  # Usa datos historicos para valores iniciales de alpha/beta
        'optimized': True,  # Optimiza automaticamente alpha y beta via MLE
    },
    'damped': {
        'initialization_method': 'estimated',  # Usa datos historicos
        'trend': 'add',  # Tendencia aditiva (lineal, no multiplicativa)
        'damped_trend': True,  # Activa phi (amortiguacion) para horizontes largos
        'seasonal': None,  # Sin componente estacional (commodities no son estacionales)
        'optimized': True,  # Optimiza alpha, beta, phi automaticamente
    }
}

# ============================================================================
# ARIMA
# ============================================================================

ARIMA_CONFIG = {
    'auto': {
        # Rango de busqueda para auto_arima
        'start_p': 0,  # Minimo orden AR
        'max_p': 5,  # Maximo orden AR (evita overfitting)
        'start_q': 0,  # Minimo orden MA
        'max_q': 5,  # Maximo orden MA
        'd': None,  # Auto-detectar orden de diferenciacion con test ADF
        'max_d': 2,  # Maximo 2 diferenciaciones
        
        # Estacionalidad
        'seasonal': False,  # Commodities no tienen estacionalidad clara
        
        # Optimizacion
        'stepwise': True,  # Busqueda heuristica rapida (vs grid search completo)
        'information_criterion': 'aic',  # Penaliza complejidad del modelo
        'maxiter': 200,  # Iteraciones maximas para convergencia
        'suppress_warnings': True,  # Evita warnings por modelos no convergentes
        'error_action': 'ignore',  # Ignora modelos que fallan (prueba siguiente)
    }
}

# ============================================================================
# ARIMAX (con variables exogenas)
# ============================================================================

ARIMAX_CONFIG = {
    'auto': {
        # Rango de busqueda (igual que ARIMA)
        'start_p': 0,
        'max_p': 5,
        'start_q': 0,
        'max_q': 5,
        'd': None,
        'max_d': 2,
        'seasonal': False,
        'stepwise': True,
        'information_criterion': 'aic',
        'maxiter': 200,
        'suppress_warnings': True,
        'error_action': 'ignore',
    },
    
    # Variables exogenas por metal (cuando esten disponibles)
    'exogenous_vars': {
        'cobre': [
            'usd_index',  # Indice dolar (inversa correlacion)
            'china_pmi',  # PMI manufacturero China (demanda)
            'copper_stocks',  # Inventarios LME
        ],
        'oro': [
            'usd_index',  # Indice dolar
            'real_rates',  # Tasas de interes reales (inversa)
            'vix',  # Volatilidad (safe haven)
        ],
        'plata': [
            'usd_index',
            'gold_price',  # Correlacion con oro
            'industrial_production',  # Uso industrial
        ],
        'cobalto': [
            'ev_sales',  # Ventas vehiculos electricos
            'battery_demand',  # Demanda baterias
            'nickel_price',  # Correlacion con niquel
        ]
    },
    
    # Configuracion de fallback si no hay datos exogenos
    'fallback_to_arima': True,  # Si faltan exogenas, usar ARIMA simple
}

# ============================================================================
# PROPHET (CORREGIDO para datos mensuales)
# ============================================================================

PROPHET_CONFIG = {
    'base': {
        # Estacionalidad
        'yearly_seasonality': True,  # Detecta ciclos anuales (ej: demanda industrial)
        'weekly_seasonality': False,  # Desactivado: datos son mensuales
        'daily_seasonality': False,  # Desactivado: datos son mensuales
        'seasonality_mode': 'multiplicative',  # Para precios (cambios porcentuales)
        
        # Changepoints (puntos de cambio estructural)
        'n_changepoints': 10,  # Reducido de 25 para evitar sobreajuste
        'changepoint_prior_scale': 0.03,  # Conservador (menor = menos flexible)
        
        # Intervalos de confianza
        'interval_width': 0.95,  # Bandas al 95% (P2.5 - P97.5)
        'uncertainty_samples': 1000,  # Simulaciones para incertidumbre
    },
    
    # Regresores externos (similar a ARIMAX pero en Prophet)
    'regressors': {
        'cobre': ['usd_index', 'china_pmi'],
        'oro': ['usd_index', 'real_rates'],
        'plata': ['usd_index', 'gold_price'],
        'cobalto': ['ev_sales', 'battery_demand']
    },
}

# ============================================================================
# MONTE CARLO
# ============================================================================

MONTE_CARLO_CONFIG = {
    'n_simulations': 10000,  # 10,000 simulaciones (balance precision/tiempo)
    'confidence_levels': [0.05, 0.50, 0.95],  # Percentiles P5, P50 (mediana), P95
    'distribution': 'normal',  # Distribucion normal para shocks estocasticos
    'random_seed': 42,  # Semilla para reproducibilidad
}

# ============================================================================
# METADATOS DEL PROYECTO
# ============================================================================

PROJECT_INFO = {
    'version': '2.0.0',  # Version refactorizada
    'metals': ['cobre', 'oro', 'plata', 'cobalto'],  # Commodities a pronosticar
    'data_frequency': 'monthly',  # Frecuencia mensual de datos
    'horizon_years': 7,  # Horizonte de proyeccion: 2026-2032
    'horizon_months': 84,  # 7 anos x 12 meses
}

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

TRAIN_TEST_SPLIT = {
    'train_ratio': 0.8,  # 80% datos para entrenamiento
    'test_ratio': 0.2,  # 20% datos para validacion (backtesting)
}
