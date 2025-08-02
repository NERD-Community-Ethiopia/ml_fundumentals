"""
Data analysis visualization functions.
Provides plotting utilities for exploring and understanding the dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def create_data_analysis_plots(data: pd.DataFrame, save_path: Path) -> None:
    """
    Create comprehensive data analysis visualizations.
    
    Args:
        data: Input dataset
        save_path: Path to save visualizations
    """
    logger.info("Creating data analysis visualizations...")
    
    # 1. Target Distribution
    create_target_distribution_plot(data, save_path)
    
    # 2. Feature Distributions
    create_feature_distributions_plot(data, save_path)
    
    # 3. Correlation Matrix
    create_correlation_matrix_plot(data, save_path)
    
    # 4. Feature Importance (if available)
    create_feature_importance_plot(data, save_path)
    
    logger.info("Data analysis visualizations created successfully")

def create_target_distribution_plot(data: pd.DataFrame, save_path: Path) -> None:
    """Create target variable distribution plot."""
    plt.figure(figsize=(10, 6))
    
    # Count plot for target distribution
    target_counts = data['Diagnosis'].value_counts()
    colors = ['lightblue', 'lightcoral']
    
    bars = plt.bar(target_counts.index, target_counts.values, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, count in zip(bars, target_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Target Variable Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Diagnosis (0: Benign, 1: Malignant)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks([0, 1], ['Benign (0)', 'Malignant (1)'])
    
    # Add percentage labels
    total = len(data)
    for i, count in enumerate(target_counts.values):
        percentage = (count / total) * 100
        plt.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path / 'target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_distributions_plot(data: pd.DataFrame, save_path: Path) -> None:
    """Create feature distributions plot."""
    # Select numerical features (exclude target)
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'Diagnosis' in numerical_features:
        numerical_features.remove('Diagnosis')
    
    # Limit to first 9 features for readability
    features_to_plot = numerical_features[:9]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(features_to_plot):
        if i < len(axes):
            # Create histogram with different colors for each class
            data[data['Diagnosis'] == 0][feature].hist(alpha=0.7, bins=20, 
                                                      color='lightblue', ax=axes[i], label='Benign')
            data[data['Diagnosis'] == 1][feature].hist(alpha=0.7, bins=20, 
                                                      color='lightcoral', ax=axes[i], label='Malignant')
            
            axes[i].set_title(f'{feature}', fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
    
    # Hide empty subplots
    for i in range(len(features_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions by Diagnosis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_matrix_plot(data: pd.DataFrame, save_path: Path) -> None:
    """Create correlation matrix heatmap."""
    # Select numerical features
    numerical_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_data.corr()
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_plot(data: pd.DataFrame, save_path: Path) -> None:
    """Create feature importance plot (placeholder for now)."""
    # This would typically be populated with actual feature importance scores
    # from a trained model. For now, we'll create a placeholder.
    
    plt.figure(figsize=(10, 6))
    
    # Example feature importance (this would come from model.feature_importances_)
    features = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'Diagnosis' in features:
        features.remove('Diagnosis')
    
    # Placeholder importance scores (random for demonstration)
    np.random.seed(42)
    importance_scores = np.random.rand(len(features))
    importance_scores = importance_scores / importance_scores.sum()  # Normalize
    
    # Sort by importance
    feature_importance = list(zip(features, importance_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    features_sorted, scores_sorted = zip(*feature_importance[:10])  # Top 10
    
    bars = plt.barh(range(len(features_sorted)), scores_sorted, color='skyblue', alpha=0.8)
    plt.yticks(range(len(features_sorted)), features_sorted)
    
    plt.title('Feature Importance (Example)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close() 