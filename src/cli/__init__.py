"""
Módulo CLI con scripts ejecutables.
"""
from .run_baseline import run_all_metals
from .run_sensitivity_analysis import main as run_sensitivity_main

__all__ = ['run_all_metals', 'run_sensitivity_main']
