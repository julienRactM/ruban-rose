#!/usr/bin/env python3
"""
Confusion Matrix Generator for Breast Cancer Detection
Creates a professional confusion matrix visualization with customizable default values
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from typing import Optional, List, Tuple

def generate_confusion_matrix(
    true_positives: int = 1283,   # Cancer correctly classified as Cancer
    false_negatives: int = 194,   # Cancer misclassified as Healthy
    false_positives: int = 236,   # Healthy misclassified as Cancer
    true_negatives: int = 919,    # Healthy correctly classified as Healthy
    class_names: List[str] = ["Cancer", "Healthy"],
    title: str = "Breast Cancer Detection - Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show_metrics: bool = True
) -> None:
    """
    Generate and display a confusion matrix with medical evaluation metrics
    
    Args:
        true_positives: Cancer samples correctly classified (TP)
        false_negatives: Cancer samples misclassified as healthy (FN) 
        false_positives: Healthy samples misclassified as cancer (FP)
        true_negatives: Healthy samples correctly classified (TN)
        class_names: Names for the classes
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
        show_metrics: Whether to display detailed metrics
    """
    
    # Create confusion matrix array
    # Format: [[TP, FN], [FP, TN]]
    # Row 0: Actual Cancer, Row 1: Actual Healthy
    # Col 0: Predicted Cancer, Col 1: Predicted Healthy
    cm = np.array([
        [true_positives, false_negatives],   # Actual Cancer
        [false_positives, true_negatives]    # Actual Healthy
    ])
    
    # Calculate medical metrics
    total_samples = cm.sum()
    
    # Sensitivity (Recall) - True Positive Rate
    sensitivity = true_positives / (true_positives + false_negatives)
    
    # Specificity - True Negative Rate  
    specificity = true_negatives / (true_negatives + false_positives)
    
    # Precision - Positive Predictive Value
    precision = true_positives / (true_positives + false_positives)
    
    # Accuracy
    accuracy = (true_positives + true_negatives) / total_samples
    
    # F1-Score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    # Matthews Correlation Coefficient (MCC)
    numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    denominator = np.sqrt(
        (true_positives + false_positives) * 
        (true_positives + false_negatives) * 
        (true_negatives + false_positives) * 
        (true_negatives + false_negatives)
    )
    mcc = numerator / denominator if denominator != 0 else 0.0
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=[f'Predicted\n{name}' for name in class_names],
        yticklabels=[f'Actual\n{name}' for name in class_names],
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Number of Samples'},
        annot_kws={'size': 16, 'weight': 'bold'}
    )
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=14, fontweight='bold')
    
    # Add percentage annotations
    cm_percentages = cm.astype('float') / cm.sum() * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm_percentages[i, j]:.1f}%)', 
                    ha='center', va='center', fontsize=12, style='italic')
    
    # Add metrics text box if requested
    if show_metrics:
        metrics_text = f"""Medical Evaluation Metrics:
        
Sensitivity (Recall): {sensitivity:.3f}
Specificity: {specificity:.3f}
Precision: {precision:.3f}
Accuracy: {accuracy:.3f}
F1-Score: {f1_score:.3f}
MCC: {mcc:.3f}

Total Samples: {total_samples:,}"""
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    # Display the plot
    plt.show()
    
    # Print detailed metrics
    if show_metrics:
        print("\n" + "="*60)
        print("BREAST CANCER DETECTION - CONFUSION MATRIX ANALYSIS")
        print("="*60)
        print(f"True Positives (Cancer → Cancer):     {true_positives:,}")
        print(f"False Negatives (Cancer → Healthy):   {false_negatives:,}")
        print(f"False Positives (Healthy → Cancer):   {false_positives:,}")
        print(f"True Negatives (Healthy → Healthy):   {true_negatives:,}")
        print(f"Total Samples:                         {total_samples:,}")
        print("-"*60)
        print("MEDICAL METRICS:")
        print(f"Sensitivity (Recall):                  {sensitivity:.3f} ({sensitivity*100:.1f}%)")
        print(f"Specificity:                           {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"Precision:                             {precision:.3f} ({precision*100:.1f}%)")
        print(f"Accuracy:                              {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"F1-Score:                              {f1_score:.3f}")
        print(f"Matthews Correlation Coefficient:      {mcc:.3f}")
        print("="*60)

def create_sample_data_from_confusion_matrix(
    true_positives: int = 1283,
    false_negatives: int = 194, 
    false_positives: int = 236,
    true_negatives: int = 919
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample y_true and y_pred arrays from confusion matrix values
    Useful for testing other sklearn metrics functions
    """
    
    # Create arrays
    y_true = []
    y_pred = []
    
    # Add True Positives (actual=0/cancer, predicted=0/cancer)
    y_true.extend([0] * true_positives)
    y_pred.extend([0] * true_positives)
    
    # Add False Negatives (actual=0/cancer, predicted=1/healthy)
    y_true.extend([0] * false_negatives)
    y_pred.extend([1] * false_negatives)
    
    # Add False Positives (actual=1/healthy, predicted=0/cancer)
    y_true.extend([1] * false_positives)
    y_pred.extend([0] * false_positives)
    
    # Add True Negatives (actual=1/healthy, predicted=1/healthy)
    y_true.extend([1] * true_negatives)
    y_pred.extend([1] * true_negatives)
    
    return np.array(y_true), np.array(y_pred)

if __name__ == "__main__":
    # Example usage with your default values
    print("Generating Confusion Matrix with Default Values...")
    
    # Generate the confusion matrix
    generate_confusion_matrix(
        true_positives=1283,    # Cancer correctly identified
        false_negatives=194,    # Cancer missed (dangerous!)
        false_positives=236,    # Healthy misidentified as cancer
        true_negatives=919,     # Healthy correctly identified
        title="Breast Cancer Detection - Confusion Matrix",
        save_path="confusion_matrix_breast_cancer.png",
        show_metrics=True
    )
    
    # Optional: Create sample data for further analysis
    y_true, y_pred = create_sample_data_from_confusion_matrix(1283, 194, 236, 919)
    print(f"\nSample data created: {len(y_true):,} samples")
    print(f"Class distribution - Cancer: {np.sum(y_true == 0):,}, Healthy: {np.sum(y_true == 1):,}")