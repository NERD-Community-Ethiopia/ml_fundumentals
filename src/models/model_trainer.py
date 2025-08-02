import logging
import joblib
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training and evaluation class with MLOps best practices."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = Path(config['model']['model_path'])
        self.checkpoint_path = Path(config['model']['checkpoint_path'])
        self.production_path = Path(config['model']['production_path'])
        
        # Create directories if they don't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.production_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training with train/validation/test split."""
        # Separate features and target
        X = data.drop('Diagnosis', axis=1)
        y = data['Diagnosis']
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )
        
        # Further split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config['training']['validation_size'],
            random_state=self.config['training']['random_state'],
            stratify=y_train
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model with cross-validation."""
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config['training']['random_state'],
            n_jobs=-1
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
    
    def evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance and store metrics."""
        # Validation metrics
        y_val_pred = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        val_accuracy = self.model.score(X_val, y_val)
        val_auc = roc_auc_score(y_val, y_val_proba)
        
        # Test metrics
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        test_accuracy = self.model.score(X_test, y_test)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Store metrics
        self.metrics = {
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'cv_mean': 0.0,  # Will be updated from training
            'cv_std': 0.0
        }
        
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation AUC: {val_auc:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")
        
        return self.metrics
    
    def save_model(self, model_name: str = "breast_cancer_model") -> None:
        """Save model, scaler, and metadata."""
        # Save model
        model_file = self.checkpoint_path / f"{model_name}.joblib"
        joblib.dump(self.model, model_file)
        
        # Save scaler
        scaler_file = self.checkpoint_path / f"{model_name}_scaler.joblib"
        joblib.dump(self.scaler, scaler_file)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': type(self.model).__name__,
            'metrics': self.metrics,
            'config': self.config,
            'feature_names': list(self.model.feature_names_in_) if hasattr(self.model, 'feature_names_in_') else []
        }
        
        metadata_file = self.checkpoint_path / f"{model_name}_metadata.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Scaler saved to {scaler_file}")
        logger.info(f"Metadata saved to {metadata_file}")
    
    def create_visualizations(self, X_test: np.ndarray, y_test: np.ndarray, 
                            y_test_pred: np.ndarray, y_test_proba: np.ndarray) -> None:
        """Create and save model evaluation visualizations."""
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.checkpoint_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_importance = pd.DataFrame({
                'feature': self.model.feature_names_in_,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importances')
            plt.savefig(self.checkpoint_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Model visualizations saved") 