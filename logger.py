# logger.py
"""
Logging configuration for Multi-Lingual Travel Assistant
Provides structured logging for debugging and monitoring
"""

import logging
import sys
from datetime import datetime

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name (usually __name__ from calling module)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Create a default logger for module-level logging
default_logger = setup_logger("travel_assistant", "INFO")