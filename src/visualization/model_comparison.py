"""
Model comparison visualization functions.
Provides plotting utilities for comparing different AI models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def create_model_comparison_plots(metrics: Dict[str, Dict[str, float]], 
                                model_configs: Dict[str, Dict[str, Any]],
                                save_path: Path) -> None:
    """
    Create comprehensive model comparison visualizations.
    
    Args:
        metrics: Dictionary of model metrics
        model_configs: Dictionary of model configurations
        save_path: Path to save visualizations
    """
    logger.info("Creating model comparison visualizations...")
    
    # Filter out models with errors
    valid_metrics = {k: v for k, v in metrics.items() if 'error' not in v}
    
    if not valid_metrics:
        logger.warning("No valid metrics found for visualization")
        return
    
    # Extract data for plotting
    model_names = []
    accuracies = []
    aucs = []
    colors = []
    
    for model_name, metric_data in valid_metrics.items():
        if model_name in model_configs:
            model_names.append(model_configs[model_name]['name'])
            accuracies.append(metric_data.get('test_accuracy', 0))
            aucs.append(metric_data.get('test_auc', 0))
            colors.append(model_configs[model_name].get('color', 'blue'))
    
    # 1. Accuracy Comparison
    create_accuracy_comparison_plot(model_names, accuracies, colors, save_path)
    
    # 2. AUC Comparison
    create_auc_comparison_plot(model_names, aucs, colors, save_path)
    
    # 3. Combined Performance Plot
    create_combined_performance_plot(model_names, accuracies, aucs, colors, save_path)
    
    # 4. Model Performance Summary
    create_performance_summary_table(valid_metrics, model_configs, save_path)
    
    logger.info("Model comparison visualizations created successfully")

def create_accuracy_comparison_plot(model_names: List[str], accuracies: List[float], 
                                  colors: List[str], save_path: Path) -> None:
    """Create accuracy comparison bar plot."""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8)
    
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
    plt.savefig(save_path / 'model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_auc_comparison_plot(model_names: List[str], aucs: List[float], 
                              colors: List[str], save_path: Path) -> None:
    """Create AUC comparison bar plot."""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, aucs, color=colors, alpha=0.8)
    
    plt.title('Model AUC Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Test AUC', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / 'model_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_performance_plot(model_names: List[str], accuracies: List[float], 
                                   aucs: List[float], colors: List[str], save_path: Path) -> None:
    """Create combined accuracy and AUC comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy subplot
    bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8)
    ax1.set_title('Test Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # AUC subplot
    bars2 = ax2.bar(model_names, aucs, color=colors, alpha=0.8)
    ax2.set_title('Test AUC', fontweight='bold')
    ax2.set_ylabel('AUC')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, auc in zip(bars2, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_summary_table(metrics: Dict[str, Dict[str, float]], 
                                   model_configs: Dict[str, Dict[str, Any]], 
                                   save_path: Path) -> None:
    """Create a performance summary table visualization."""
    # This could be enhanced to create a proper table visualization
    # For now, we'll create a simple text summary
    summary_data = []
    
    for model_name, metric_data in metrics.items():
        if model_name in model_configs:
            summary_data.append({
                'Model': model_configs[model_name]['name'],
                'Accuracy': f"{metric_data.get('test_accuracy', 0):.3f}",
                'AUC': f"{metric_data.get('test_auc', 0):.3f}",
                'Description': model_configs[model_name].get('description', '')
            })
    
    # Save summary as text file
    summary_file = save_path / 'model_performance_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for data in summary_data:
            f.write(f"Model: {data['Model']}\n")
            f.write(f"Accuracy: {data['Accuracy']}\n")
            f.write(f"AUC: {data['AUC']}\n")
            f.write(f"Description: {data['Description']}\n")
            f.write("-" * 30 + "\n\n") 