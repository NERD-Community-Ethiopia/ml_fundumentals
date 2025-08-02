import logging
import joblib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import json
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MultiModelServing:
    """Multi-model serving system for different datasets."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.model_path = Path(model_path)
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.feature_names = {}
        
        # Load all available models
        self._load_all_models()
    
    def _load_all_models(self) -> None:
        """Load all available models from the checkpoint directory."""
        logger.info("Loading all available models...")
        
        # Find all model files
        model_files = glob.glob(str(self.model_path / "checkpoints" / "*_model_*.joblib"))
        
        for model_file in model_files:
            try:
                # Extract dataset name from filename
                filename = Path(model_file).name
                dataset_name = filename.split('_model_')[0]
                
                # Load model
                model = joblib.load(model_file)
                self.models[dataset_name] = model
                
                # Try to load corresponding scaler
                scaler_file = model_file.replace('_model_', '_scaler_')
                if Path(scaler_file).exists():
                    self.scalers[dataset_name] = joblib.load(scaler_file)
                
                # Try to load metadata
                metadata_file = model_file.replace('.joblib', '_metadata.yaml')
                if Path(metadata_file).exists():
                    with open(metadata_file, 'r') as f:
                        self.metadata[dataset_name] = yaml.safe_load(f)
                        # Get feature names from metadata
                        feature_names = self.metadata[dataset_name].get('feature_names', [])
                        if feature_names:
                            self.feature_names[dataset_name] = feature_names
                        else:
                            # Fallback: use default feature names for breast cancer dataset
                            self.feature_names[dataset_name] = [
                                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                                'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                                'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                                'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                                'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
                            ]
                else:
                    # No metadata file found, use default feature names
                    self.feature_names[dataset_name] = [
                        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
                    ]
                
                logger.info(f"Loaded {dataset_name} model successfully")
                
            except Exception as e:
                logger.error(f"Error loading model from {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def preprocess_input(self, data: Union[pd.DataFrame, Dict, List], dataset_name: str) -> np.ndarray:
        """Preprocess input data for a specific model."""
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input must be dict, list, or DataFrame")
        
        # Get feature names for this dataset
        if dataset_name not in self.feature_names:
            raise ValueError(f"No feature names found for dataset: {dataset_name}")
        
        required_features = self.feature_names[dataset_name]
        
        # Ensure all required features are present
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features for {dataset_name}: {missing_features}")
        
        # Select only required features in correct order
        df = df[required_features]
        
        # Scale features if scaler exists
        if dataset_name in self.scalers:
            df_scaled = self.scalers[dataset_name].transform(df)
        else:
            df_scaled = df.values
        
        return df_scaled
    
    def predict_single_model(self, data: Union[pd.DataFrame, Dict, List], dataset_name: str) -> Dict[str, Any]:
        """Make predictions using a single model."""
        if dataset_name not in self.models:
            raise ValueError(f"Model not found: {dataset_name}")
        
        try:
            # Preprocess input
            X = self.preprocess_input(data, dataset_name)
            
            # Make predictions
            model = self.models[dataset_name]
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Format results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'prediction': int(pred),
                    'probability_benign': float(prob[0]),
                    'probability_malignant': float(prob[1]),
                    'confidence': float(max(prob)),
                    'prediction_label': 'Malignant' if pred == 1 else 'Benign',
                    'model_used': dataset_name
                }
                results.append(result)
            
            return {
                'predictions': results,
                'model_info': {
                    'model_type': type(model).__name__,
                    'dataset': dataset_name,
                    'version': self.metadata.get(dataset_name, {}).get('timestamp', 'Unknown'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {dataset_name}: {e}")
            raise
    
    def predict_all_models(self, data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
        """Make predictions using all available models."""
        all_predictions = {}
        
        for dataset_name in self.models.keys():
            try:
                prediction = self.predict_single_model(data, dataset_name)
                all_predictions[dataset_name] = prediction
            except Exception as e:
                logger.error(f"Error predicting with {dataset_name} model: {e}")
                all_predictions[dataset_name] = {'error': str(e)}
        
        return {
            'all_predictions': all_predictions,
            'timestamp': datetime.now().isoformat(),
            'models_used': list(self.models.keys())
        }
    
    def compare_predictions(self, data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
        """Compare predictions across all models."""
        all_predictions = self.predict_all_models(data)
        
        # Extract predictions for comparison
        comparison = {
            'input_data': data if isinstance(data, dict) else data[0] if isinstance(data, list) else data.iloc[0].to_dict(),
            'model_predictions': {},
            'agreement': True,
            'consensus_prediction': None,
            'timestamp': datetime.now().isoformat()
        }
        
        predictions = []
        for dataset_name, result in all_predictions['all_predictions'].items():
            if 'error' not in result:
                pred = result['predictions'][0]['prediction']
                confidence = result['predictions'][0]['confidence']
                predictions.append(pred)
                
                comparison['model_predictions'][dataset_name] = {
                    'prediction': pred,
                    'prediction_label': 'Malignant' if pred == 1 else 'Benign',
                    'confidence': confidence,
                    'probability_benign': result['predictions'][0]['probability_benign'],
                    'probability_malignant': result['predictions'][0]['probability_malignant']
                }
        
        # Check if all models agree
        if len(set(predictions)) > 1:
            comparison['agreement'] = False
        
        # Get consensus prediction (majority vote)
        if predictions:
            consensus = max(set(predictions), key=predictions.count)
            comparison['consensus_prediction'] = {
                'prediction': consensus,
                'prediction_label': 'Malignant' if consensus == 1 else 'Benign',
                'agreement_count': predictions.count(consensus),
                'total_models': len(predictions)
            }
        
        return comparison
    
    def get_model_info(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about models."""
        if dataset_name:
            if dataset_name not in self.metadata:
                return {'error': f'Model {dataset_name} not found'}
            
            return {
                'model_type': self.metadata[dataset_name].get('model_type', 'Unknown'),
                'version': self.metadata[dataset_name].get('timestamp', 'Unknown'),
                'features': self.feature_names.get(dataset_name, []),
                'feature_count': len(self.feature_names.get(dataset_name, [])),
                'metrics': self.metadata[dataset_name].get('metrics', {}),
                'dataset_info': self.metadata[dataset_name].get('dataset_info', {}),
                'last_updated': self.metadata[dataset_name].get('timestamp', 'Unknown')
            }
        else:
            # Return info for all models
            all_info = {}
            for name in self.models.keys():
                all_info[name] = self.get_model_info(name)
            return all_info
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models."""
        health_status = {
            'status': 'healthy',
            'models_loaded': len(self.models),
            'model_status': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for dataset_name, model in self.models.items():
            try:
                # Test with dummy data
                if dataset_name in self.feature_names:
                    dummy_data = {feature: 0.0 for feature in self.feature_names[dataset_name]}
                    prediction = self.predict_single_model(dummy_data, dataset_name)
                    
                    health_status['model_status'][dataset_name] = {
                        'status': 'healthy',
                        'model_loaded': True,
                        'scaler_loaded': dataset_name in self.scalers,
                        'metadata_loaded': dataset_name in self.metadata,
                        'prediction_test': 'passed'
                    }
                else:
                    health_status['model_status'][dataset_name] = {
                        'status': 'warning',
                        'error': 'No feature names available'
                    }
                    
            except Exception as e:
                health_status['model_status'][dataset_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'unhealthy'
        
        return health_status
    
    def save_prediction_log(self, input_data: Dict, prediction: Dict, 
                          output_path: str = "logs/multi_model_predictions.jsonl") -> None:
        """Log predictions for monitoring and debugging."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': prediction,
            'models_used': list(self.models.keys())
        }
        
        # Ensure log directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Append to log file
        with open(output_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"Multi-model prediction logged to {output_path}")

class MultiModelAPI:
    """API wrapper for multi-model serving."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.multi_model_serving = MultiModelServing(model_path, config)
    
    def predict_single_endpoint(self, data: Dict, dataset_name: str) -> Dict[str, Any]:
        """API endpoint for single model prediction."""
        try:
            result = self.multi_model_serving.predict_single_model(data, dataset_name)
            self.multi_model_serving.save_prediction_log(data, result)
            return {
                'status': 'success',
                'data': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_all_endpoint(self, data: Dict) -> Dict[str, Any]:
        """API endpoint for all models prediction."""
        try:
            result = self.multi_model_serving.predict_all_models(data)
            self.multi_model_serving.save_prediction_log(data, result)
            return {
                'status': 'success',
                'data': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def compare_endpoint(self, data: Dict) -> Dict[str, Any]:
        """API endpoint for prediction comparison."""
        try:
            result = self.multi_model_serving.compare_predictions(data)
            return {
                'status': 'success',
                'data': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def health_endpoint(self) -> Dict[str, Any]:
        """API endpoint for health check."""
        return self.multi_model_serving.health_check()
    
    def models_info_endpoint(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """API endpoint for model information."""
        try:
            info = self.multi_model_serving.get_model_info(dataset_name)
            return {
                'status': 'success',
                'data': info
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def available_models_endpoint(self) -> Dict[str, Any]:
        """API endpoint to get available models."""
        return {
            'status': 'success',
            'data': {
                'available_models': self.multi_model_serving.get_available_models(),
                'total_models': len(self.multi_model_serving.models)
            }
        } 

# Create FastAPI app
app = FastAPI(
    title="Multi-Model AI Comparison API",
    description="API for comparing different AI models on the same dataset",
    version="1.0.0"
)

# Pydantic models for API
class PredictionRequest(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

# Global API instance
api_instance = None

def initialize_api():
    """Initialize the API with models."""
    global api_instance
    if api_instance is None:
        try:
            import yaml
            with open("configs/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            api_instance = MultiModelAPI('models', config)
            logger.info("API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API: {e}")
            raise

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    initialize_api()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Model AI Comparison API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")
    return api_instance.health_endpoint()

@app.get("/models")
async def get_models():
    """Get available models."""
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")
    return api_instance.available_models_endpoint()

@app.post("/predict/{model_name}")
async def predict_single(model_name: str, data: PredictionRequest):
    """Predict using a single model."""
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")
    return api_instance.predict_single_endpoint(data.dict(), model_name)

@app.post("/predict/all")
async def predict_all(data: PredictionRequest):
    """Predict using all models."""
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")
    return api_instance.predict_all_endpoint(data.dict())

@app.post("/compare")
async def compare_predictions(data: PredictionRequest):
    """Compare predictions across all models."""
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")
    return api_instance.compare_endpoint(data.dict()) 