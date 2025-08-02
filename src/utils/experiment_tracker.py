import logging
import mlflow
import mlflow.sklearn
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import wandb

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Experiment tracking and model versioning with MLflow and W&B."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = config.get('experiment', {}).get('name', 'breast_cancer_classification')
        self.tracking_uri = config.get('experiment', {}).get('tracking_uri', 'sqlite:///mlruns.db')
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        # Initialize W&B if configured
        self.use_wandb = config.get('experiment', {}).get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config.get('experiment', {}).get('wandb_project', 'breast-cancer-ml'),
                config=config
            )
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")
        return run
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow and W&B."""
        mlflow.log_params(params)
        
        if self.use_wandb:
            wandb.config.update(params)
        
        logger.info(f"Logged parameters: {params}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow and W&B."""
        mlflow.log_metrics(metrics, step=step)
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        logger.info(f"Logged metrics: {metrics}")
    
    def log_model(self, model, model_name: str = "breast_cancer_model") -> None:
        """Log model to MLflow."""
        mlflow.sklearn.log_model(model, model_name)
        logger.info(f"Logged model: {model_name}")
    
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts to MLflow and W&B."""
        mlflow.log_artifacts(local_path, artifact_path)
        
        if self.use_wandb:
            wandb.save(local_path)
        
        logger.info(f"Logged artifacts from: {local_path}")
    
    def log_data_version(self, data_path: str, data_hash: str) -> None:
        """Log data version information."""
        data_info = {
            'data_path': data_path,
            'data_hash': data_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        mlflow.log_dict(data_info, "data_version.json")
        
        if self.use_wandb:
            wandb.log({"data_hash": data_hash})
        
        logger.info(f"Logged data version: {data_hash}")
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info("Ended MLflow run")
    
    def get_best_run(self, metric: str = "test_accuracy", mode: str = "max") -> Optional[Dict[str, Any]]:
        """Get the best run based on a metric."""
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=[f"metrics.{metric} DESC" if mode == "max" else f"metrics.{metric} ASC"]
            )
            
            if not runs.empty:
                best_run = runs.iloc[0]
                return {
                    'run_id': best_run['run_id'],
                    'metrics': {k: v for k, v in best_run.items() if k.startswith('metrics.')},
                    'params': {k: v for k, v in best_run.items() if k.startswith('params.')}
                }
        except Exception as e:
            logger.error(f"Error getting best run: {e}")
        
        return None
    
    def load_model(self, run_id: str, model_name: str = "breast_cancer_model"):
        """Load a model from MLflow."""
        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from run: {run_id}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

class ModelMonitor:
    """Model monitoring and drift detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.drift_threshold = config.get('monitoring', {}).get('drift_threshold', 0.1)
        self.monitoring_path = Path(config.get('monitoring', {}).get('monitoring_path', 'logs/monitoring'))
        self.monitoring_path.mkdir(parents=True, exist_ok=True)
    
    def calculate_data_drift(self, reference_data, current_data, feature_names: list) -> Dict[str, float]:
        """Calculate data drift between reference and current data."""
        drift_scores = {}
        
        for feature in feature_names:
            if feature in reference_data.columns and feature in current_data.columns:
                # Calculate distribution difference (simplified)
                ref_mean = reference_data[feature].mean()
                ref_std = reference_data[feature].std()
                curr_mean = current_data[feature].mean()
                curr_std = current_data[feature].std()
                
                # Simple drift score based on mean and std differences
                mean_diff = abs(ref_mean - curr_mean) / (ref_std + 1e-8)
                std_diff = abs(ref_std - curr_std) / (ref_std + 1e-8)
                drift_scores[feature] = (mean_diff + std_diff) / 2
        
        return drift_scores
    
    def detect_drift(self, drift_scores: Dict[str, float]) -> Dict[str, Any]:
        """Detect if drift exceeds threshold."""
        high_drift_features = {
            feature: score for feature, score in drift_scores.items() 
            if score > self.drift_threshold
        }
        
        alert = len(high_drift_features) > 0
        
        return {
            'alert': alert,
            'high_drift_features': high_drift_features,
            'drift_threshold': self.drift_threshold,
            'timestamp': datetime.now().isoformat()
        }
    
    def log_monitoring_data(self, monitoring_data: Dict[str, Any]) -> None:
        """Log monitoring data to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.monitoring_path / f"monitoring_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(monitoring_data, f, indent=2)
        
        logger.info(f"Monitoring data logged to {log_file}")
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate a monitoring report."""
        # This would typically aggregate monitoring data over time
        # For now, return a simple structure
        return {
            'total_alerts': 0,
            'last_check': datetime.now().isoformat(),
            'drift_threshold': self.drift_threshold,
            'status': 'healthy'
        } 