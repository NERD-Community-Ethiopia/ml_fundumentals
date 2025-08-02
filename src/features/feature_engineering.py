import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering and selection class."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_names = []
        self.selected_features = []
        self.scaler = None
        self.feature_selector = None
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        df = data.copy()
        
        # Create statistical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'Diagnosis']
        
        if len(numerical_cols) >= 2:
            # Create ratio features
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    ratio_name = f"ratio_{col1}_{col2}"
                    df[ratio_name] = df[col1] / (df[col2] + 1e-8)  # Avoid division by zero
            
            # Create statistical aggregations
            df['mean_features'] = df[numerical_cols].mean(axis=1)
            df['std_features'] = df[numerical_cols].std(axis=1)
            df['max_features'] = df[numerical_cols].max(axis=1)
            df['min_features'] = df[numerical_cols].min(axis=1)
            
            # Create polynomial features for top features
            top_features = numerical_cols[:5]  # Use first 5 features
            for col in top_features:
                df[f"{col}_squared"] = df[col] ** 2
                df[f"{col}_cubed"] = df[col] ** 3
        
        logger.info(f"Created {len(df.columns) - len(data.columns)} new features")
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'kbest', n_features: int = 20) -> pd.DataFrame:
        """Select the most important features."""
        if method == 'kbest':
            # Use statistical tests
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_indices = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_indices].tolist()
            
        elif method == 'rfe':
            # Use Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(estimator=estimator, n_features_to_select=n_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.support_].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features using {method}")
        logger.info(f"Selected features: {self.selected_features[:10]}...")  # Show first 10
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def scale_features(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale features using specified method."""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get feature importance scores."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_dict = dict(zip(X.columns, rf.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_feature_info(self, output_path: str) -> None:
        """Save feature engineering information."""
        feature_info = {
            'selected_features': self.selected_features,
            'total_features': len(self.selected_features),
            'scaling_method': type(self.scaler).__name__ if self.scaler else None,
            'feature_selector': type(self.feature_selector).__name__ if self.feature_selector else None
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(feature_info, f, default_flow_style=False)
        
        logger.info(f"Feature information saved to {output_path}")
    
    def validate_features(self, data: pd.DataFrame) -> bool:
        """Validate feature quality and data integrity."""
        issues = []
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            issues.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            issues.append(f"Infinite values found: {inf_counts[inf_counts > 0].to_dict()}")
        
        # Check for constant features
        constant_features = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"Constant features found: {constant_features}")
        
        # Check for highly correlated features
        corr_matrix = data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        if high_corr_features:
            issues.append(f"Highly correlated features (>0.95): {high_corr_features}")
        
        if issues:
            logger.warning("Feature validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        
        logger.info("Feature validation passed")
        return True 