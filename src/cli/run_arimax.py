"""
Pipeline completo ARIMAX: combinar datos + entrenar + evaluar
"""

import sys
from pathlib import Path

# Agregar raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.merge_exogenous import prepare_all_metals
from src.models.arimax_model import run_arimax_experiment

if __name__ == '__main__':
    # Paso 1: Combinar metales con exógenas
    print("="*60)
    print("PASO 1: Combinando datos con variables exógenas")
    print("="*60)
    datasets = prepare_all_metals()
    
    # Paso 2: Ejecutar ARIMAX para cada metal
    print("\n" + "="*60)
    print("PASO 2: Ejecutando modelos ARIMAX")
    print("="*60)
    
    metals = ['cobre', 'oro', 'plata', 'cobalto']
    results = {}
    
    for metal in metals:
        result = run_arimax_experiment(metal, test_size=24)
        if result:
            results[metal] = result
    
    print("\nPipeline ARIMAX completado")
