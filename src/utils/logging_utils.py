"""
Logging configuration for the Medical Diagnosis AI System
"""

import logging
import os
from datetime import datetime

def setup_logging():
    """
    Sets up a centralized logger for the application.
    
    - Creates a 'logs' directory if it doesn't exist.
    - Configures a logger to write to a timestamped file in the 'logs' directory.
    - Returns the configured logger instance.
    """
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file name with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"diagnosis_run_{timestamp}.log")
    
    # Get the root logger
    logger = logging.getLogger("MedicalDiagnosisAgentSystem")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger

# Initialize a global logger
logger = setup_logging()
