#!/usr/bin/env python3
"""
Multi-Model AI Comparison Pipeline for Breast Cancer Classification.
This demonstrates how to compare different AI algorithms on the same dataset.
Perfect for teaching students about model selection and comparison.
"""

import logging
import yaml
from pathlib import Path
import pandas as pd

# Import our modules
from models.multi_model_trainer import MultiModelTrainer
from models.multi_model_serving import MultiModelServing, MultiModelAPI
from utils.experiment_tracker import ExperimentTracker, ModelMonitor

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

def run_multi_model_training(config: dict, use_experiment_tracking: bool = False):
    """Run training with multiple AI models on the same dataset."""
    logger.info("Starting multi-model AI comparison pipeline...")
    
    # Initialize experiment tracking if requested
    experiment_tracker = None
    if use_experiment_tracking:
        try:
            experiment_tracker = ExperimentTracker(config)
            run = experiment_tracker.start_run("multi_model_comparison")
            logger.info("✅ Experiment tracking enabled")
        except Exception as e:
            logger.warning(f"⚠️  Experiment tracking failed: {e}. Continuing without it.")
            use_experiment_tracking = False
    
    trainer = MultiModelTrainer(config)
    data_path = 'data/raw/data.csv'  # Using Kaggle format
    
    logger.info("Training multiple AI models on the same dataset...")
    
    # Log parameters if experiment tracking is enabled
    if use_experiment_tracking and experiment_tracker:
        experiment_tracker.log_parameters({
            'data_path': data_path,
            'models': list(trainer.model_configs.keys()),
            'test_size': config['training']['test_size'],
            'validation_size': config['training']['validation_size']
        })
    
    all_metrics = trainer.train_all_models(data_path)
    
    # Log metrics for each model if experiment tracking is enabled
    if use_experiment_tracking and experiment_tracker:
        for model_name, metrics in all_metrics.items():
            if 'error' not in metrics:
                experiment_tracker.log_metrics({
                    f'{model_name}_test_accuracy': metrics['test_accuracy'],
                    f'{model_name}_test_auc': metrics['test_auc'],
                    f'{model_name}_val_accuracy': metrics['val_accuracy'],
                    f'{model_name}_val_auc': metrics['val_auc']
                })
    
    ensemble = trainer.create_ensemble_model()
    trainer.save_models()
    
    # Use enhanced visualization module if available
    try:
        from visualization.model_comparison import create_model_comparison_plots
        logger.info("Using enhanced visualization module...")
        create_model_comparison_plots(trainer.metrics, trainer.model_configs, trainer.checkpoint_path)
    except ImportError:
        logger.info("Using built-in visualization...")
        trainer.create_comparison_visualizations()
    
    summary = trainer.get_model_summary()
    
    # End experiment tracking if enabled
    if use_experiment_tracking and experiment_tracker:
        experiment_tracker.end_run()
        logger.info("✅ Experiment tracking completed")
    
    logger.info("Multi-model training completed!")
    return summary

def test_multi_model_serving(config: dict):
    """Test the multi-model serving system."""
    logger.info("Testing multi-model serving system...")
    
    # Initialize multi-model serving
    serving = MultiModelServing('models', config)
    
    # Get available models
    available_models = serving.get_available_models()
    logger.info(f"Available models: {available_models}")
    
    # Create sample test data
    if available_models:
        # Use features from the breast cancer dataset
        sample_data = {
            'radius_mean': 17.99,
            'texture_mean': 10.38,
            'perimeter_mean': 122.8,
            'area_mean': 1001.0,
            'smoothness_mean': 0.1184,
            'compactness_mean': 0.2776,
            'concavity_mean': 0.3001,
            'concave points_mean': 0.1471,
            'symmetry_mean': 0.2419,
            'fractal_dimension_mean': 0.07871,
            'radius_se': 1.095,
            'texture_se': 0.9053,
            'perimeter_se': 8.589,
            'area_se': 153.4,
            'smoothness_se': 0.006399,
            'compactness_se': 0.04904,
            'concavity_se': 0.05373,
            'concave points_se': 0.01587,
            'symmetry_se': 0.03003,
            'fractal_dimension_se': 0.006193,
            'radius_worst': 25.38,
            'texture_worst': 17.33,
            'perimeter_worst': 184.6,
            'area_worst': 2019.0,
            'smoothness_worst': 0.1622,
            'compactness_worst': 0.6656,
            'concavity_worst': 0.7119,
            'concave points_worst': 0.2654,
            'symmetry_worst': 0.4601,
            'fractal_dimension_worst': 0.1189
        }
        
        logger.info("Testing predictions with sample data...")
        
        # Test predictions from different models
        for model_name in available_models[:3]:  # Test first 3 models
            try:
                pred = serving.predict_single_model(sample_data, model_name)
                logger.info(f"{model_name}: {pred['predictions'][0]['prediction_label']} (confidence: {pred['predictions'][0]['confidence']:.3f})")
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
        
        # Test all models prediction
        try:
            all_preds = serving.predict_all_models(sample_data)
            logger.info("All models predictions:")
            for model_name, result in all_preds['all_predictions'].items():
                if 'error' not in result:
                    pred_label = result['predictions'][0]['prediction_label']
                    confidence = result['predictions'][0]['confidence']
                    logger.info(f"  {model_name}: {pred_label} (confidence: {confidence:.3f})")
        except Exception as e:
            logger.error(f"Error in all models prediction: {e}")
    
    # Test health check
    health = serving.health_check()
    logger.info(f"System health: {health['status']}")
    logger.info(f"Models loaded: {health['models_loaded']}")
    
    return serving

def print_teaching_summary(summary: dict):
    """Print a teaching-focused summary of the results."""
    logger.info("\n" + "="*80)
    logger.info("TEACHING SUMMARY: Multi-Model AI Comparison")
    logger.info("="*80)
    
    logger.info("\n🎯 LEARNING OBJECTIVES:")
    logger.info("1. Compare different AI algorithms on the same dataset")
    logger.info("2. Understand model strengths and weaknesses")
    logger.info("3. Learn about ensemble methods")
    logger.info("4. Practice model selection and evaluation")
    
    logger.info("\n📊 MODELS TRAINED:")
    for model_name in summary['models_trained']:
        if model_name in summary['model_metrics'] and 'error' not in summary['model_metrics'][model_name]:
            metrics = summary['model_metrics'][model_name]
            logger.info(f"  • {metrics['model_name']}: {metrics['test_accuracy']:.3f} accuracy, {metrics['test_auc']:.3f} AUC")
    
    logger.info(f"\n🏆 BEST PERFORMING MODEL: {summary['best_performing_model']}")
    
    logger.info("\n💡 KEY INSIGHTS:")
    logger.info("• Different algorithms have different strengths")
    logger.info("• Performance varies based on data characteristics")
    logger.info("• Ensemble methods can improve overall performance")
    logger.info("• Consider interpretability vs. performance trade-offs")
    
    logger.info("\n📁 GENERATED FILES:")
    logger.info("• Model files: models/checkpoints/")
    logger.info("• Comparison reports: models/checkpoints/model_comparison_report_*.yaml")
    logger.info("• Visualizations: models/checkpoints/*_comparison.png")
    
    logger.info("\n🚀 NEXT STEPS:")
    logger.info("1. Analyze the comparison reports")
    logger.info("2. Study the visualizations")
    logger.info("3. Experiment with hyperparameter tuning")
    logger.info("4. Try different ensemble combinations")

def main():
    """Main function orchestrating the multi-model AI comparison pipeline."""
    logger.info("Starting Multi-Model AI Comparison Pipeline for Teaching")
    
    try:
        config = load_config()
        
        # Check if experiment tracking is requested via environment or config
        use_experiment_tracking = config.get('experiment', {}).get('use_experiment_tracking', False)
        
        if use_experiment_tracking:
            logger.info("🚀 ADVANCED MODE: Running with experiment tracking")
            summary = run_multi_model_training(config, use_experiment_tracking=True)
        else:
            logger.info("📚 TEACHING MODE: Running simple pipeline")
            summary = run_multi_model_training(config, use_experiment_tracking=False)
        
        serving = test_multi_model_serving(config)
        print_teaching_summary(summary)
        
        logger.info("🎉 MULTI-MODEL AI COMPARISON PIPELINE COMPLETED!")
        logger.info("=" * 80)
        logger.info("This demonstrates how to compare different AI algorithms")
        logger.info("on the same dataset - a crucial skill in machine learning!")
        logger.info("")
        logger.info("Check the generated files for detailed analysis.")
        
        # Show mode-specific next steps
        if use_experiment_tracking:
            logger.info("🔬 ADVANCED FEATURES:")
            logger.info("• Check MLflow UI: mlflow ui")
            logger.info("• View experiment history in mlruns/")
            logger.info("• Compare runs and track model versions")
        else:
            logger.info("📖 TEACHING FEATURES:")
            logger.info("• Simple and easy to understand")
            logger.info("• Focus on model comparison concepts")
            logger.info("• Ready for classroom use")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 