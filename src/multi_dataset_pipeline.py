#!/usr/bin/env python3
"""
Multi-dataset training and serving pipeline for breast cancer classification.
Trains models on both Kaggle and Wisconsin datasets and provides comparison.
"""

import logging
import yaml
from pathlib import Path
import pandas as pd

# Import our modules
from models.multi_dataset_trainer import MultiDatasetTrainer
from models.multi_model_serving import MultiModelServing, MultiModelAPI

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

def run_multi_dataset_training(config: dict):
    """Run training on multiple datasets."""
    logger.info("Starting multi-dataset training pipeline...")
    
    # Initialize multi-dataset trainer
    trainer = MultiDatasetTrainer(config)
    
    # Define datasets
    datasets = {
        'kaggle': 'data/raw/data.csv',
        'wisconsin': 'data/raw/breast+cancer+wisconsin+diagnostic/wdbc.data'
    }
    
    # Train models on all datasets
    logger.info("Training models on all datasets...")
    all_metrics = trainer.train_all_models(datasets)
    
    # Create ensemble model
    logger.info("Creating ensemble model...")
    ensemble = trainer.create_ensemble_model()
    
    # Save all models
    logger.info("Saving all models...")
    trainer.save_models()
    
    # Create comparison visualizations
    logger.info("Creating comparison visualizations...")
    trainer.create_comparison_visualizations()
    
    # Get summary
    summary = trainer.get_model_summary()
    
    logger.info("Multi-dataset training completed!")
    logger.info(f"Trained {summary['total_models']} models: {summary['datasets']}")
    logger.info(f"Best performing model: {summary['best_performing_model']}")
    
    return summary

def test_multi_model_serving(config: dict):
    """Test the multi-model serving system."""
    logger.info("Testing multi-model serving system...")
    
    # Initialize multi-model serving
    serving = MultiModelServing('models', config)
    
    # Get available models
    available_models = serving.get_available_models()
    logger.info(f"Available models: {available_models}")
    
    # Create sample test data (using features from first available model)
    if available_models:
        first_model = available_models[0]
        feature_names = serving.feature_names.get(first_model, [])
        
        if feature_names:
            # Create sample data with all features set to 0
            test_data = {feature: 0.0 for feature in feature_names}
            
            logger.info("Testing predictions with sample data...")
            
            # Test single model prediction
            try:
                single_pred = serving.predict_single_model(test_data, first_model)
                logger.info(f"Single model prediction ({first_model}): {single_pred['predictions'][0]['prediction_label']}")
            except Exception as e:
                logger.error(f"Error in single model prediction: {e}")
            
            # Test all models prediction
            try:
                all_preds = serving.predict_all_models(test_data)
                logger.info("All models predictions:")
                for model_name, result in all_preds['all_predictions'].items():
                    if 'error' not in result:
                        pred_label = result['predictions'][0]['prediction_label']
                        confidence = result['predictions'][0]['confidence']
                        logger.info(f"  {model_name}: {pred_label} (confidence: {confidence:.3f})")
            except Exception as e:
                logger.error(f"Error in all models prediction: {e}")
            
            # Test comparison
            try:
                comparison = serving.compare_predictions(test_data)
                logger.info(f"Model agreement: {comparison['agreement']}")
                if comparison['consensus_prediction']:
                    consensus = comparison['consensus_prediction']['prediction_label']
                    agreement_count = comparison['consensus_prediction']['agreement_count']
                    total_models = comparison['consensus_prediction']['total_models']
                    logger.info(f"Consensus prediction: {consensus} ({agreement_count}/{total_models} models agree)")
            except Exception as e:
                logger.error(f"Error in comparison: {e}")
    
    # Test health check
    health = serving.health_check()
    logger.info(f"System health: {health['status']}")
    logger.info(f"Models loaded: {health['models_loaded']}")
    
    return serving

def main():
    """Main function orchestrating the multi-dataset pipeline."""
    logger.info("Starting multi-dataset breast cancer classification pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Project: {config['project']['name']}")
        
        # Step 1: Train models on multiple datasets
        summary = run_multi_dataset_training(config)
        
        # Step 2: Test multi-model serving
        serving = test_multi_model_serving(config)
        
        # Step 3: Print final summary
        logger.info("\n" + "="*60)
        logger.info("MULTI-DATASET PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Models trained: {summary['datasets']}")
        logger.info(f"Best model: {summary['best_performing_model']}")
        logger.info(f"Available for serving: {serving.get_available_models()}")
        logger.info("\nNext steps:")
        logger.info("1. Use MultiModelAPI for serving")
        logger.info("2. Access comparison reports in models/checkpoints/")
        logger.info("3. View visualizations for model comparison")
        logger.info("4. Deploy with Docker for production")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 