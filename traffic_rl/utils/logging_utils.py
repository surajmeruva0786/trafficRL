"""
Logging utilities for training and evaluation.

This module provides functions for setting up loggers and
logging episode information.
"""

import logging
import os
import csv
from typing import Dict, Any, List
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, console only)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_episode(
    episode: int,
    metrics: Dict[str, Any],
    logger: logging.Logger = None
) -> None:
    """
    Log episode metrics.
    
    Args:
        episode: Episode number
        metrics: Dictionary of metrics
        logger: Logger to use (if None, print to console)
    """
    message = f"Episode {episode}: "
    message += ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                         for k, v in metrics.items()])
    
    if logger:
        logger.info(message)
    else:
        print(message)


class CSVLogger:
    """
    CSV logger for saving metrics to file.
    """
    
    def __init__(self, filepath: str, fieldnames: List[str]):
        """
        Initialize CSV logger.
        
        Args:
            filepath: Path to CSV file
            fieldnames: List of column names
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create file and write header
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def log(self, data: Dict[str, Any]) -> None:
        """
        Log a row of data.
        
        Args:
            data: Dictionary of data to log
        """
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)
    
    def log_batch(self, data_list: List[Dict[str, Any]]) -> None:
        """
        Log multiple rows of data.
        
        Args:
            data_list: List of dictionaries to log
        """
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(data_list)


def create_run_directory(base_dir: str, prefix: str = "run") -> str:
    """
    Create a timestamped run directory.
    
    Args:
        base_dir: Base directory for runs
        prefix: Prefix for run directory name
    
    Returns:
        Path to created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test_logger", "test.log")
    logger.info("This is a test message")
    
    # Test CSV logger
    csv_logger = CSVLogger("test_metrics.csv", ["episode", "reward", "loss"])
    csv_logger.log({"episode": 1, "reward": -100.5, "loss": 0.5})
    csv_logger.log({"episode": 2, "reward": -95.2, "loss": 0.45})
    
    print("Logging test complete")
