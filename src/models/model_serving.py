import logging
import joblib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ModelServing:
    """Model serving class for production deployment."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = []
        
        # Load model components
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model, scaler, and metadata."""
        try:
            # Load model
            model_file = self.model_path / "breast_cancer_model.joblib"
            self.model = joblib.load(model_file)
            
            # Load scaler
            scaler_file = self.model_path / "breast_cancer_model_scaler.joblib"
            self.scaler = joblib.load(scaler_file)
            
            # Load metadata
            metadata_file = self.model_path / "breast_cancer_model_metadata.yaml"
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.safe_load(f)
            
            self.feature_names = self.metadata.get('feature_names', [])
            
            logger.info("Model components loaded successfully")
            logger.info(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
            logger.info(f"Features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, data: Union[pd.DataFrame, Dict, List]) -> np.ndarray:
        """Preprocess input data for prediction."""
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input must be dict, list, or DataFrame")
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only required features in correct order
        df = df[self.feature_names]
        
        # Scale features
        if self.scaler:
            df_scaled = self.scaler.transform(df)
        else:
            df_scaled = df.values
        
        return df_scaled
    
    def predict(self, data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
        """Make predictions on input data."""
        try:
            # Preprocess input
            X = self.preprocess_input(data)
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Format results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'prediction': int(pred),
                    'probability_benign': float(prob[0]),
                    'probability_malignant': float(prob[1]),
                    'confidence': float(max(prob)),
                    'prediction_label': 'Malignant' if pred == 1 else 'Benign'
                }
                results.append(result)
            
            return {
                'predictions': results,
                'model_info': {
                    'model_type': self.metadata.get('model_type', 'Unknown'),
                    'version': self.metadata.get('version', '1.0.0'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        return {
            'model_type': self.metadata.get('model_type', 'Unknown'),
            'version': self.metadata.get('version', '1.0.0'),
            'features': self.feature_names,
            'feature_count': len(self.feature_names),
            'metrics': self.metadata.get('metrics', {}),
            'training_config': self.metadata.get('config', {}),
            'last_updated': self.metadata.get('timestamp', 'Unknown')
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model."""
        try:
            # Test with dummy data
            dummy_data = {feature: 0.0 for feature in self.feature_names}
            prediction = self.predict(dummy_data)
            
            return {
                'status': 'healthy',
                'model_loaded': True,
                'scaler_loaded': self.scaler is not None,
                'metadata_loaded': self.metadata is not None,
                'prediction_test': 'passed',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_predict(self, data: List[Dict]) -> Dict[str, Any]:
        """Make batch predictions."""
        try:
            df = pd.DataFrame(data)
            return self.predict(df)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def save_prediction_log(self, input_data: Dict, prediction: Dict, 
                          output_path: str = "logs/predictions.jsonl") -> None:
        """Log predictions for monitoring and debugging."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': prediction,
            'model_version': self.metadata.get('version', '1.0.0')
        }
        
        # Ensure log directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Append to log file
        with open(output_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"Prediction logged to {output_path}")

# Example usage for API endpoints
class ModelAPI:
    """Simple API wrapper for model serving."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_serving = ModelServing(model_path, config)
    
    def predict_endpoint(self, data: Dict) -> Dict[str, Any]:
        """API endpoint for single prediction."""
        try:
            result = self.model_serving.predict(data)
            self.model_serving.save_prediction_log(data, result)
            return {
                'status': 'success',
                'data': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def batch_predict_endpoint(self, data: List[Dict]) -> Dict[str, Any]:
        """API endpoint for batch predictions."""
        try:
            result = self.model_serving.batch_predict(data)
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
        return self.model_serving.health_check()
    
    def info_endpoint(self) -> Dict[str, Any]:
        """API endpoint for model information."""
        return {
            'status': 'success',
            'data': self.model_serving.get_model_info()
        } 