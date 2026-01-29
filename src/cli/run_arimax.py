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
    print("="*60)
    print("PIPELINE COMPLETO ARIMAX")
    print("="*60)
    
    # Paso 1: Verificar/Combinar datos con exógenas
    print("\nPASO 1: Preparando datos con variables exógenas")
    print("-"*60)
    
    # Verificar si ya existen los datos procesados
    data_dir = Path('data/processed')
    metals = ['cobre', 'oro', 'plata', 'cobalto']
    all_exist = all((data_dir / f'{m}_with_exogenous.csv').exists() for m in metals)
    
    if all_exist:
        print("[OK] Datos procesados ya existen. Saltando preparación.")
        print("     (Si quieres regenerarlos, borra data/processed/*.csv)")
    else:
        print("[INFO] Generando datos procesados...")
        datasets = prepare_all_metals()
        if not datasets:
            print("[ERROR] No se pudieron preparar los datos")
            sys.exit(1)
        print("[OK] Datos procesados generados")
    
    # Paso 2: Ejecutar ARIMAX para cada metal (80/20 split)
    print("\nPASO 2: Ejecutando modelos ARIMAX (80% train / 20% test)")
    print("-"*60)
    
    results = {}
    
    for metal in metals:
        try:
            print(f"\n>>> Procesando {metal.upper()}...")
            result = run_arimax_experiment(metal, test_ratio=0.2)
            if result:
                results[metal] = result
                print(f"[OK] {metal.upper()} completado")
            else:
                print(f"[ERROR] {metal.upper()} fallo")
        except Exception as e:
            print(f"[ERROR] en {metal}: {e}")
            continue
    
    # Resumen final
    print("\n" + "="*60)
    print("PIPELINE COMPLETADO")
    print("="*60)
    
    if results:
        print(f"\n[OK] {len(results)}/{len(metals)} metales procesados exitosamente")
        print(f"     Resultados en: results/arimax/")
    else:
        print("\n[ERROR] No se generaron resultados")
