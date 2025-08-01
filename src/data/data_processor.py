import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the DataProcessor class (copied from your input)
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

# Step 1: Define configuration
config = {
    'data': {
        'raw_data_path': 'data/raw',  # Directory where wdbc.data is stored
        'processed_data_path': 'data/processed'  # Directory to save processed data
    }
}

# Step 2: Initialize DataProcessor
processor = DataProcessor(config)

# Step 3: Load data
filename = 'data.csv'
data = processor.load_data(filename)

if data is not None:
    # Step 4: Preprocess data
    # Check actual number of columns and adjust column names accordingly
    num_columns = len(data.columns)
    logger.info(f"Data has {num_columns} columns")
    logger.info(f"Data shape: {data.shape}")
    
    # Add column names (since wdbc.data has no header)
    # The Wisconsin Breast Cancer dataset typically has 32 columns: ID, Diagnosis, and 30 features
    if num_columns == 32:
        column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    else:
        # Fallback for different column counts
        column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, num_columns - 1)]
    
    data.columns = column_names
    
    # Convert Diagnosis to binary (M=1, B=0)
    data['Diagnosis'] = (data['Diagnosis'] == 'M').astype(int)
    
    # Drop ID column (not useful for analysis)
    data = data.drop('ID', axis=1)
    
    # Basic preprocessing (e.g., check for missing values)
    logger.info("Missing Values:\n%s", data.isnull().sum())
    
    # Step 5: Save processed data
    processed_filename = 'breast_cancer_processed.csv'
    success = processor.save_data(data, processed_filename)
    if success:
        logger.info("Processed data saved successfully.")
    
    # Step 6: Visualize data (example: distribution of Feature_1)
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='Feature_1', hue='Diagnosis', kde=True, bins=30)
    plt.title('Distribution of Feature_1 (Mean Radius) by Diagnosis')
    plt.xlabel('Mean Radius')
    plt.ylabel('Count')
    plt.legend(labels=['Benign', 'Malignant'])
    plt.savefig('data/processed/feature1_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation matrix heatmap (example visualization)
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.drop('Diagnosis', axis=1).corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Features')
    plt.savefig('data/processed/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved to data/processed/")
else:
    logger.error("Failed to load data. Check file path or format.")