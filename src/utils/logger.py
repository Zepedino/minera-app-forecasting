"""
Sistema de logging centralizado.

Registra informacion en consola y archivo simultaneamente.
"""

import logging
from datetime import datetime
from config import LOGS_DIR


def setup_logger(name="minera_forecasting", level=logging.INFO):
    """
    Configura logger con salida a consola y archivo.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evitar duplicar handlers si se llama multiples veces
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Formato de mensajes
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (un archivo por dia)
    log_filename = LOGS_DIR / f"forecasting_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
