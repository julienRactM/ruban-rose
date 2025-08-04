#!/usr/bin/env python3
"""
Test script to verify AUC-ROC fix works correctly
"""

import numpy as np
import sys
import os

# Add src to path to import trainer
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training.trainer import MedicalMetrics

def test_auc_roc_fix():
    """Test that the AUC-ROC calculation fix works correctly"""
    
    print("=== Testing AUC-ROC Fix ===")
    
    # Create test data based on your actual confusion matrix
    # Validation Confusion Matrix from logs:
    # Cancer:     [  5184] [    249]
    # Healthy:    [   871] [   1589]
    
    np.random.seed(42)
    
    # Build test arrays
    y_true = []
    y_pred = []
    y_proba = []
    
    # True Positives: Cancer correctly predicted (5184 samples)
    for _ in range(5184):
        y_true.append(0)  # Actual cancer
        y_pred.append(0)  # Predicted cancer
        cancer_prob = np.random.beta(8, 2)  # High cancer probability
        y_proba.append([cancer_prob, 1 - cancer_prob])
    
    # False Negatives: Cancer incorrectly predicted as healthy (249 samples)
    for _ in range(249):
        y_true.append(0)  # Actual cancer
        y_pred.append(1)  # Predicted healthy
        cancer_prob = np.random.beta(2, 8)  # Low cancer probability (mistake)
        y_proba.append([cancer_prob, 1 - cancer_prob])
    
    # False Positives: Healthy incorrectly predicted as cancer (871 samples)
    for _ in range(871):
        y_true.append(1)  # Actual healthy
        y_pred.append(0)  # Predicted cancer
        cancer_prob = np.random.beta(8, 2)  # High cancer probability (mistake)
        y_proba.append([cancer_prob, 1 - cancer_prob])
    
    # True Negatives: Healthy correctly predicted (1589 samples)
    for _ in range(1589):
        y_true.append(1)  # Actual healthy
        y_pred.append(1)  # Predicted healthy
        cancer_prob = np.random.beta(2, 8)  # Low cancer probability
        y_proba.append([cancer_prob, 1 - cancer_prob])
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    print(f"Test data created:")
    print(f"  Total samples: {len(y_true)}")
    print(f"  Cancer samples: {np.sum(y_true == 0)}")
    print(f"  Healthy samples: {np.sum(y_true == 1)}")
    print(f"  y_proba shape: {y_proba.shape}")
    
    # Calculate metrics using the fixed function
    print(f"\n=== Using Fixed MedicalMetrics ===")
    metrics = MedicalMetrics.calculate_metrics(y_true, y_pred, y_proba)
    
    # Print key metrics
    print(f"Sensitivity (Cancer recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity (Healthy recall): {metrics['specificity']:.4f}")
    print(f"AUC-ROC (FIXED): {metrics['auc_roc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    
    # Verify confusion matrix matches expectation
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix Verification:")
    print(f"Expected vs Actual:")
    print(f"Cancer:  [5184, 249] vs [{cm[0,0]:4d}, {cm[0,1]:3d}]")
    print(f"Healthy: [871, 1589] vs [{cm[1,0]:4d}, {cm[1,1]:4d}]")
    
    # Check if AUC-ROC is now reasonable
    if metrics['auc_roc'] > 0.7:
        print(f"\n✅ SUCCESS: AUC-ROC is now {metrics['auc_roc']:.4f} (reasonable value)")
        print("   The fix worked! Your models should now show proper AUC-ROC values.")
    elif metrics['auc_roc'] < 0.3:
        print(f"\n❌ ISSUE: AUC-ROC is still {metrics['auc_roc']:.4f} (too low)")
        print("   Something is still wrong with the calculation.")
    else:
        print(f"\n⚠️  BORDERLINE: AUC-ROC is {metrics['auc_roc']:.4f}")
        print("   This might indicate model performance issues rather than calculation errors.")
    
    # Test edge case: all probabilities equal (should give AUC = 0.5)
    print(f"\n=== Edge Case Test: Random Classifier ===")
    y_proba_random = np.full((len(y_true), 2), 0.5)  # All 50% probabilities
    y_pred_random = np.random.choice([0, 1], size=len(y_true))  # Random predictions
    
    metrics_random = MedicalMetrics.calculate_metrics(y_true, y_pred_random, y_proba_random)
    print(f"Random classifier AUC-ROC: {metrics_random['auc_roc']:.4f}")
    
    if abs(metrics_random['auc_roc'] - 0.5) < 0.1:
        print("✅ Edge case passed: Random classifier gives ~0.5 AUC-ROC")
    else:
        print(f"❌ Edge case failed: Expected ~0.5, got {metrics_random['auc_roc']:.4f}")

if __name__ == "__main__":
    test_auc_roc_fix()