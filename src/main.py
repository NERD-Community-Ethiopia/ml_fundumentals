#!/usr/bin/env python3
"""
Main entry point for the ML project with full MLOps pipeline.
"""

import logging
import yaml
from pathlib import Path
import pandas as pd

# Import our modules
from data.data_processor import DataProcessor
from features.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_data_pipeline(config: dict) -> pd.DataFrame:
    """Run the complete data processing pipeline."""
    logger.info("Starting data processing pipeline...")
    
    # Initialize data processor
    processor = DataProcessor(config)
    
    # Load raw data
    filename = 'data.csv'
    data = processor.load_data(filename)
    
    if data is None:
        raise ValueError("Failed to load data")
    
    # Basic preprocessing
    num_columns = len(data.columns)
    logger.info(f"Data has {num_columns} columns")
    
    # Add column names
    if num_columns == 32:
        column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    else:
        column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, num_columns - 1)]
    
    data.columns = column_names
    
    # Convert Diagnosis to binary
    data['Diagnosis'] = (data['Diagnosis'] == 'M').astype(int)
    
    # Drop ID column
    data = data.drop('ID', axis=1)
    
    # Save processed data
    processed_filename = 'breast_cancer_processed.csv'
    success = processor.save_data(data, processed_filename)
    
    if not success:
        raise ValueError("Failed to save processed data")
    
    logger.info("Data processing pipeline completed")
    return data

def run_feature_engineering(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline...")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Validate features
    if not feature_engineer.validate_features(data):
        logger.warning("Feature validation failed, but continuing...")
    
    # Create new features
    data_with_features = feature_engineer.create_features(data)
    
    # Separate features and target
    X = data_with_features.drop('Diagnosis', axis=1)
    y = data_with_features['Diagnosis']
    
    # Select best features
    X_selected = feature_engineer.select_features(X, y, method='kbest', n_features=20)
    
    # Scale features
    X_scaled = feature_engineer.scale_features(X_selected, method='standard')
    
    # Combine with target
    final_data = pd.concat([X_scaled, y], axis=1)
    
    # Save feature information
    feature_engineer.save_feature_info('models/feature_info.yaml')
    
    logger.info("Feature engineering pipeline completed")
    return final_data

def run_model_training(data: pd.DataFrame, config: dict):
    """Run the model training pipeline."""
    logger.info("Starting model training pipeline...")
    
    # Initialize model trainer
    trainer = ModelTrainer(config)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(data)
    
    # Train model
    trainer.train_model(X_train, y_train)
    
    # Evaluate model
    metrics = trainer.evaluate_model(X_val, y_val, X_test, y_test)
    
    # Save model
    trainer.save_model("breast_cancer_model")
    
    # Create visualizations
    y_test_pred = trainer.model.predict(X_test)
    y_test_proba = trainer.model.predict_proba(X_test)[:, 1]
    trainer.create_visualizations(X_test, y_test, y_test_pred, y_test_proba)
    
    logger.info("Model training pipeline completed")
    return metrics

def main():
    """Main function orchestrating the complete MLOps pipeline."""
    logger.info("Starting ML project with MLOps pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Project: {config['project']['name']}")
        
        # Step 1: Data Processing
        data = run_data_pipeline(config)
        
        # Step 2: Feature Engineering
        processed_data = run_feature_engineering(data, config)
        
        # Step 3: Model Training
        metrics = run_model_training(processed_data, config)
        
        # Log final results
        logger.info("MLOps pipeline completed successfully!")
        logger.info(f"Final model metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
