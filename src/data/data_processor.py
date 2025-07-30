"""
Data processing utilities.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing class."""
    
    def __init__(self, config: dict):
        self.config = config
        self.raw_data_path = Path(config['data']['raw_data_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])
        
    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from file."""
        try:
            file_path = self.raw_data_path / filename
            if file_path.suffix == '.csv':
                return pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def save_data(self, data: pd.DataFrame, filename: str) -> bool:
        """Save processed data."""
        try:
            file_path = self.processed_data_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
