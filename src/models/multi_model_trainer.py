import logging
import joblib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb

logger = logging.getLogger(__name__)

class MultiModelTrainer:
    """Multi-model training and comparison system for different AI algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = Path(config['model']['model_path'])
        self.checkpoint_path = Path(config['model']['checkpoint_path'])
        self.production_path = Path(config['model']['production_path'])
        
        # Create directories
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.production_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models and scalers
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.model_configs = {}
        
        # Define different model configurations
        self._define_model_configs()
    
    def _define_model_configs(self):
        """Define different AI model configurations for comparison."""
        self.model_configs = {
            'random_forest': {
                'name': 'Random Forest',
                'description': 'Ensemble of decision trees, good for non-linear relationships',
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': 50,  # Reduced for speed
                    'max_depth': 8,
                    'random_state': self.config['training']['random_state'],
                    'n_jobs': -1
                },
                'color': 'skyblue'
            },
            'logistic_regression': {
                'name': 'Logistic Regression',
                'description': 'Linear model, interpretable and fast',
                'model_class': LogisticRegression,
                'params': {
                    'random_state': self.config['training']['random_state'],
                    'max_iter': 500,  # Reduced for speed
                    'solver': 'liblinear'
                },
                'color': 'orange',
                'needs_scaling': True
            },
            'xgboost': {
                'name': 'XGBoost',
                'description': 'Optimized gradient boosting, often best performance',
                'model_class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 50,  # Reduced for speed
                    'learning_rate': 0.1,
                    'max_depth': 4,
                    'random_state': self.config['training']['random_state']
                },
                'color': 'red'
            }
        }
    
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the breast cancer dataset."""
        logger.info(f"Loading dataset from: {file_path}")
        
        # Load data
        data = pd.read_csv(file_path)
        
        # Handle different column formats
        if 'diagnosis' in data.columns:
            data = data.rename(columns={'diagnosis': 'Diagnosis'})
        elif 'Diagnosis' not in data.columns:
            # For Wisconsin format without headers
            column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
            data.columns = column_names
            data = data.drop('ID', axis=1)
        
        # Convert diagnosis to binary
        data['Diagnosis'] = (data['Diagnosis'] == 'M').astype(int)
        
        # Drop ID column if exists
        if 'id' in data.columns:
            data = data.drop('id', axis=1)
        
        # Handle NaN values - drop rows with NaN or fill with median
        logger.info(f"Checking for NaN values...")
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            logger.info(f"Found {nan_count} NaN values. Cleaning data...")
            # Drop any unnamed columns that might contain NaN
            unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
            if unnamed_cols:
                data = data.drop(columns=unnamed_cols)
                logger.info(f"Dropped unnamed columns: {unnamed_cols}")
            
            # Fill remaining NaN values with median
            data = data.fillna(data.median())
            logger.info("Filled NaN values with median")
        
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Diagnosis distribution: {data['Diagnosis'].value_counts().to_dict()}")
        logger.info(f"Features: {list(data.drop('Diagnosis', axis=1).columns)}")
        
        return data
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training, validation, and testing."""
        # Separate features and target
        X = data.drop('Diagnosis', axis=1)
        y = data['Diagnosis']
        
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        # Split data: train -> temp, temp -> val + test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['training']['validation_size'],
            random_state=self.config['training']['random_state'],
            stratify=y_temp
        )
        
        # Store training data for ensemble fitting
        self.X_train = X_train.values
        self.y_train = y_train.values
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values
    
    def train_single_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                          model_name: str, scaler: StandardScaler = None) -> Any:
        """Train a single model with optional hyperparameter tuning."""
        logger.info(f"Training {model_name} model...")
        
        config = self.model_configs[model_name]
        model_class = config['model_class']
        params = config['params'].copy()
        
        # Scale data if needed
        if config.get('needs_scaling', False) and scaler is not None:
            X_train_scaled = scaler.transform(X_train)
        else:
            X_train_scaled = X_train
        
        # Initialize model
        model = model_class(**params)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation scores for {model_name}: {cv_scores}")
        logger.info(f"CV Mean for {model_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        model.fit(X_train_scaled, y_train)
        
        # Store model
        self.models[model_name] = model
        
        return model, cv_scores
    
    def evaluate_single_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray, model_name: str,
                            scaler: StandardScaler = None) -> Dict[str, float]:
        """Evaluate a single model."""
        config = self.model_configs[model_name]
        
        # Scale data if needed
        if config.get('needs_scaling', False) and scaler is not None:
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_val_scaled = X_val
            X_test_scaled = X_test
        
        # Validation metrics
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        val_accuracy = model.score(X_val_scaled, y_val)
        val_auc = roc_auc_score(y_val, y_val_proba)
        
        # Test metrics
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        test_accuracy = model.score(X_test_scaled, y_test)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Store metrics
        metrics = {
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'model_name': config['name'],
            'description': config['description']
        }
        
        self.metrics[model_name] = metrics
        
        logger.info(f"Results for {config['name']}:")
        logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"  Validation AUC: {val_auc:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}")
        
        return metrics
    
    def train_all_models(self, data_path: str) -> Dict[str, Dict[str, float]]:
        """Train all models on the same dataset."""
        logger.info("Starting multi-model training pipeline...")
        
        # Load and preprocess data
        data = self.load_and_preprocess_data(data_path)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(data)
        
        # Create scaler for models that need it
        scaler = StandardScaler()
        scaler.fit(X_train)
        self.scalers['standard'] = scaler
        
        all_metrics = {}
        
        # Train each model
        for model_name in self.model_configs.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {self.model_configs[model_name]['name']}")
            logger.info(f"{'='*50}")
            
            try:
                # Train model
                model, cv_scores = self.train_single_model(X_train, y_train, model_name, scaler)
                
                # Evaluate model
                metrics = self.evaluate_single_model(model, X_val, y_val, X_test, y_test, model_name, scaler)
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                all_metrics[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                all_metrics[model_name] = {'error': str(e)}
        
        return all_metrics
    
    def create_ensemble_model(self) -> Any:
        """Create an ensemble model from all trained models."""
        logger.info("Creating ensemble model from all algorithms")
        
        # Get models that trained successfully
        successful_models = {name: model for name, model in self.models.items() 
                           if name in self.metrics and 'error' not in self.metrics[name]}
        
        if len(successful_models) < 2:
            logger.warning("Not enough successful models for ensemble")
            return None
        
        # Create voting classifier
        from sklearn.ensemble import VotingClassifier
        
        estimators = []
        for model_name, model in successful_models.items():
            config = self.model_configs[model_name]
            estimators.append((f'{model_name}_model', model))
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability voting
        )
        
        # Fit the ensemble with training data
        if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            ensemble.fit(self.X_train, self.y_train)
            logger.info("Ensemble model fitted with training data")
        
        # Store ensemble model
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def save_models(self) -> None:
        """Save all models, scalers, and metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get feature names from the data
        feature_names = list(self.feature_names) if hasattr(self, 'feature_names') else []
        
        for model_name, model in self.models.items():
            # Save model
            model_file = self.checkpoint_path / f"{model_name}_model_{timestamp}.joblib"
            joblib.dump(model, model_file)
            
            # Save metadata (avoid YAML serialization issues)
            metadata = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'config_name': self.model_configs.get(model_name, {}).get('name', 'Unknown'),
                'config_description': self.model_configs.get(model_name, {}).get('description', ''),
                'metrics': self.metrics.get(model_name, {}),
                'feature_names': feature_names,
                'timestamp': timestamp
            }
            
            metadata_file = self.checkpoint_path / f"{model_name}_metadata_{timestamp}.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            logger.info(f"Saved {model_name} model to {model_file}")
        
        # Save scaler
        scaler_file = self.checkpoint_path / f"scaler_{timestamp}.joblib"
        joblib.dump(self.scalers['standard'], scaler_file)
        
        # Save comparison report
        self.save_comparison_report(timestamp)
    
    def save_comparison_report(self, timestamp: str) -> None:
        """Save a comprehensive comparison report."""
        report = {
            'timestamp': timestamp,
            'model_comparison': {},
            'best_model': None,
            'recommendations': [],
            'teaching_insights': []
        }
        
        best_accuracy = 0
        best_model = None
        
        for model_name, metrics in self.metrics.items():
            if 'error' not in metrics:
                report['model_comparison'][model_name] = {
                    'name': metrics['model_name'],
                    'description': metrics['description'],
                    'test_accuracy': metrics['test_accuracy'],
                    'test_auc': metrics['test_auc'],
                    'val_accuracy': metrics['val_accuracy'],
                    'val_auc': metrics['val_auc'],
                    'cv_mean': metrics.get('cv_mean', 0),
                    'cv_std': metrics.get('cv_std', 0)
                }
                
                if metrics['test_accuracy'] > best_accuracy:
                    best_accuracy = metrics['test_accuracy']
                    best_model = model_name
        
        report['best_model'] = best_model
        
        # Generate teaching insights
        accuracies = [metrics['test_accuracy'] for metrics in self.metrics.values() if 'error' not in metrics]
        if accuracies:
            accuracy_range = max(accuracies) - min(accuracies)
            if accuracy_range < 0.05:
                report['teaching_insights'].append("Models perform similarly - consider interpretability and speed")
            else:
                report['teaching_insights'].append(f"Significant performance differences - {best_model} performs best")
        
        # Add model-specific insights
        for model_name, metrics in self.metrics.items():
            if 'error' not in metrics:
                if model_name == 'logistic_regression':
                    report['teaching_insights'].append("Logistic Regression: Good baseline, highly interpretable")
                elif model_name == 'random_forest':
                    report['teaching_insights'].append("Random Forest: Robust, handles non-linear relationships well")
                elif model_name == 'xgboost':
                    report['teaching_insights'].append("XGBoost: Often best performance, but can overfit")
                elif model_name == 'neural_network':
                    report['teaching_insights'].append("Neural Network: Complex patterns, but needs more data")
        
        # Save report
        report_file = self.checkpoint_path / f"model_comparison_report_{timestamp}.yaml"
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"Comparison report saved to {report_file}")
    
    def create_comparison_visualizations(self) -> None:
        """Create visualizations comparing model performance."""
        # Performance comparison
        model_names = []
        accuracies = []
        aucs = []
        colors = []
        
        for model_name, metrics in self.metrics.items():
            if 'error' not in metrics:
                model_names.append(self.model_configs[model_name]['name'])
                accuracies.append(metrics['test_accuracy'])
                aucs.append(metrics['test_auc'])
                colors.append(self.model_configs[model_name]['color'])
        
        # Accuracy comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color=colors)
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_path / 'model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # AUC comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, aucs, color=colors)
        plt.title('Model AUC Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Test AUC', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        for bar, auc in zip(bars, aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_path / 'model_auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cross-validation comparison
        cv_means = []
        cv_stds = []
        
        for model_name, metrics in self.metrics.items():
            if 'error' not in metrics:
                cv_means.append(metrics.get('cv_mean', 0))
                cv_stds.append(metrics.get('cv_std', 0))
        
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(model_names))
        plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color=colors, alpha=0.7)
        plt.title('Cross-Validation Performance', fontsize=14, fontweight='bold')
        plt.ylabel('CV Accuracy', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_path / 'model_cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison visualizations saved")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of all trained models."""
        summary = {
            'total_models': len(self.models),
            'models_trained': list(self.models.keys()),
            'best_performing_model': None,
            'model_metrics': self.metrics,
            'teaching_purpose': "This demonstrates how to compare different AI algorithms on the same dataset"
        }
        
        # Find best model
        best_accuracy = 0
        for model_name, metrics in self.metrics.items():
            if 'error' not in metrics and metrics['test_accuracy'] > best_accuracy:
                best_accuracy = metrics['test_accuracy']
                summary['best_performing_model'] = model_name
        
        return summary 