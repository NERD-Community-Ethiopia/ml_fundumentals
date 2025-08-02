import logging
import joblib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)

class MultiDatasetTrainer:
    """Multi-dataset training and comparison system."""
    
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
        self.dataset_info = {}
        
    def load_and_preprocess_dataset(self, dataset_name: str, file_path: str) -> pd.DataFrame:
        """Load and preprocess a specific dataset."""
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "kaggle":
            # Load Kaggle dataset
            data = pd.read_csv(file_path)
            # Rename columns to match Wisconsin format
            data = data.rename(columns={'diagnosis': 'Diagnosis'})
            # Convert diagnosis to binary
            data['Diagnosis'] = (data['Diagnosis'] == 'M').astype(int)
            # Drop ID column
            data = data.drop('id', axis=1)
            
        elif dataset_name == "wisconsin":
            # Load Wisconsin dataset
            data = pd.read_csv(file_path, header=None)
            # Add column names
            column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
            data.columns = column_names
            # Convert diagnosis to binary
            data['Diagnosis'] = (data['Diagnosis'] == 'M').astype(int)
            # Drop ID column
            data = data.drop('ID', axis=1)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        logger.info(f"Dataset {dataset_name} shape: {data.shape}")
        logger.info(f"Diagnosis distribution: {data['Diagnosis'].value_counts().to_dict()}")
        
        return data
    
    def prepare_data(self, data: pd.DataFrame, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[dataset_name] = scaler
        
        logger.info(f"Data split for {dataset_name} - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_single_model(self, X_train: np.ndarray, y_train: np.ndarray, dataset_name: str) -> RandomForestClassifier:
        """Train a model on a single dataset."""
        logger.info(f"Training model for {dataset_name} dataset")
        
        # Initialize model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config['training']['random_state'],
            n_jobs=-1
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation scores for {dataset_name}: {cv_scores}")
        logger.info(f"CV Mean for {dataset_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Store model
        self.models[dataset_name] = model
        
        return model
    
    def evaluate_single_model(self, model: RandomForestClassifier, X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Evaluate a single model."""
        # Validation metrics
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        val_accuracy = model.score(X_val, y_val)
        val_auc = roc_auc_score(y_val, y_val_proba)
        
        # Test metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        test_accuracy = model.score(X_test, y_test)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Store metrics
        metrics = {
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'cv_mean': 0.0,  # Will be updated
            'cv_std': 0.0
        }
        
        self.metrics[dataset_name] = metrics
        
        logger.info(f"Results for {dataset_name}:")
        logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"  Validation AUC: {val_auc:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}")
        
        return metrics
    
    def train_all_models(self, datasets: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Train models on all datasets."""
        all_metrics = {}
        
        for dataset_name, file_path in datasets.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'='*50}")
            
            # Load and preprocess data
            data = self.load_and_preprocess_dataset(dataset_name, file_path)
            
            # Store dataset info
            self.dataset_info[dataset_name] = {
                'shape': data.shape,
                'diagnosis_distribution': data['Diagnosis'].value_counts().to_dict(),
                'features': list(data.drop('Diagnosis', axis=1).columns)
            }
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(data, dataset_name)
            
            # Train model
            model = self.train_single_model(X_train, y_train, dataset_name)
            
            # Evaluate model
            metrics = self.evaluate_single_model(model, X_val, y_val, X_test, y_test, dataset_name)
            all_metrics[dataset_name] = metrics
        
        return all_metrics
    
    def create_ensemble_model(self) -> VotingClassifier:
        """Create an ensemble model from all trained models."""
        logger.info("Creating ensemble model from all datasets")
        
        # Create voting classifier
        estimators = []
        for dataset_name, model in self.models.items():
            estimators.append((f'{dataset_name}_model', model))
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability voting
        )
        
        # Store ensemble model
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def save_models(self) -> None:
        """Save all models, scalers, and metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for dataset_name, model in self.models.items():
            # Save model
            model_file = self.checkpoint_path / f"{dataset_name}_model_{timestamp}.joblib"
            joblib.dump(model, model_file)
            
            # Save scaler if exists
            if dataset_name in self.scalers:
                scaler_file = self.checkpoint_path / f"{dataset_name}_scaler_{timestamp}.joblib"
                joblib.dump(self.scalers[dataset_name], scaler_file)
            
            # Save metadata
            metadata = {
                'model_name': f"{dataset_name}_model",
                'model_type': type(model).__name__,
                'metrics': self.metrics.get(dataset_name, {}),
                'dataset_info': self.dataset_info.get(dataset_name, {}),
                'config': self.config,
                'timestamp': timestamp,
                'feature_names': list(self.dataset_info[dataset_name]['features']) if dataset_name in self.dataset_info else []
            }
            
            metadata_file = self.checkpoint_path / f"{dataset_name}_metadata_{timestamp}.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            logger.info(f"Saved {dataset_name} model to {model_file}")
        
        # Save comparison report
        self.save_comparison_report(timestamp)
    
    def save_comparison_report(self, timestamp: str) -> None:
        """Save a comprehensive comparison report."""
        report = {
            'timestamp': timestamp,
            'model_comparison': {},
            'dataset_info': self.dataset_info,
            'best_model': None,
            'recommendations': []
        }
        
        best_accuracy = 0
        best_model = None
        
        for dataset_name, metrics in self.metrics.items():
            report['model_comparison'][dataset_name] = {
                'test_accuracy': metrics['test_accuracy'],
                'test_auc': metrics['test_auc'],
                'val_accuracy': metrics['val_accuracy'],
                'val_auc': metrics['val_auc']
            }
            
            if metrics['test_accuracy'] > best_accuracy:
                best_accuracy = metrics['test_accuracy']
                best_model = dataset_name
        
        report['best_model'] = best_model
        
        # Generate recommendations
        accuracies = [metrics['test_accuracy'] for metrics in self.metrics.values()]
        if max(accuracies) - min(accuracies) < 0.05:
            report['recommendations'].append("Models perform similarly - ensemble recommended")
        else:
            report['recommendations'].append(f"Use {best_model} model for best performance")
        
        # Save report
        report_file = self.checkpoint_path / f"comparison_report_{timestamp}.yaml"
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"Comparison report saved to {report_file}")
    
    def create_comparison_visualizations(self) -> None:
        """Create visualizations comparing model performance."""
        # Performance comparison
        datasets = list(self.metrics.keys())
        accuracies = [self.metrics[d]['test_accuracy'] for d in datasets]
        aucs = [self.metrics[d]['test_auc'] for d in datasets]
        
        # Accuracy comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(datasets, accuracies, color=['skyblue', 'lightgreen', 'orange'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Test Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.savefig(self.checkpoint_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # AUC comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(datasets, aucs, color=['skyblue', 'lightgreen', 'orange'])
        plt.title('Model AUC Comparison')
        plt.ylabel('Test AUC')
        plt.ylim(0, 1)
        
        for bar, auc in zip(bars, aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        plt.savefig(self.checkpoint_path / 'auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Comparison visualizations saved")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of all trained models."""
        summary = {
            'total_models': len(self.models),
            'datasets': list(self.models.keys()),
            'best_performing_model': None,
            'model_metrics': self.metrics,
            'dataset_info': self.dataset_info
        }
        
        # Find best model
        best_accuracy = 0
        for dataset_name, metrics in self.metrics.items():
            if metrics['test_accuracy'] > best_accuracy:
                best_accuracy = metrics['test_accuracy']
                summary['best_performing_model'] = dataset_name
        
        return summary 