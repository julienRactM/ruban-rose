#!/usr/bin/env python3
"""
Debug script to understand AUC-ROC calculation issue
"""

import numpy as np
from sklearn.metrics import roc_auc_score

def debug_auc_roc_calculation():
    """Debug the AUC-ROC calculation issue"""
    
    print("=== AUC-ROC Calculation Debug ===")
    
    # Simulate your actual confusion matrix from the logs:
    # Validation Confusion Matrix:
    # Cancer:     [  5184] [    249]  
    # Healthy:    [   871] [   1589]  
    
    # Create ground truth labels
    cancer_correct = 5184  # True Positives
    cancer_wrong = 249     # False Negatives
    healthy_wrong = 871    # False Positives  
    healthy_correct = 1589 # True Negatives
    
    # Build y_true array
    y_true = []
    y_true.extend([0] * cancer_correct)   # Cancer samples correctly predicted
    y_true.extend([0] * cancer_wrong)     # Cancer samples incorrectly predicted  
    y_true.extend([1] * healthy_wrong)    # Healthy samples incorrectly predicted
    y_true.extend([1] * healthy_correct)  # Healthy samples correctly predicted
    
    y_true = np.array(y_true)
    
    print(f"y_true shape: {y_true.shape}")
    print(f"Cancer samples (class 0): {np.sum(y_true == 0)}")
    print(f"Healthy samples (class 1): {np.sum(y_true == 1)}")
    
    # Simulate softmax probabilities that would lead to this confusion matrix
    # For a good model, probabilities should be:
    # - High prob for class 0 when y_true=0 and prediction is correct
    # - Low prob for class 0 when y_true=0 and prediction is wrong
    # - High prob for class 0 when y_true=1 and prediction is wrong (false positive)
    # - Low prob for class 0 when y_true=1 and prediction is correct
    
    np.random.seed(42)  # For reproducibility
    
    # Create realistic probability distributions
    y_proba = np.zeros((len(y_true), 2))
    
    idx = 0
    
    # Cancer samples correctly predicted as cancer (TP) - should have high prob for class 0
    for _ in range(cancer_correct):
        prob_cancer = np.random.beta(8, 2)  # High probability for cancer
        y_proba[idx] = [prob_cancer, 1 - prob_cancer]
        idx += 1
    
    # Cancer samples incorrectly predicted as healthy (FN) - should have low prob for class 0  
    for _ in range(cancer_wrong):
        prob_cancer = np.random.beta(2, 8)  # Low probability for cancer
        y_proba[idx] = [prob_cancer, 1 - prob_cancer]
        idx += 1
        
    # Healthy samples incorrectly predicted as cancer (FP) - should have high prob for class 0
    for _ in range(healthy_wrong):
        prob_cancer = np.random.beta(8, 2)  # High probability for cancer (wrong!)
        y_proba[idx] = [prob_cancer, 1 - prob_cancer]
        idx += 1
        
    # Healthy samples correctly predicted as healthy (TN) - should have low prob for class 0
    for _ in range(healthy_correct):
        prob_cancer = np.random.beta(2, 8)  # Low probability for cancer  
        y_proba[idx] = [prob_cancer, 1 - prob_cancer]
        idx += 1
    
    print(f"y_proba shape: {y_proba.shape}")
    print(f"y_proba[:, 0] (cancer probs) range: [{y_proba[:, 0].min():.3f}, {y_proba[:, 0].max():.3f}]")
    print(f"y_proba[:, 1] (healthy probs) range: [{y_proba[:, 1].min():.3f}, {y_proba[:, 1].max():.3f}]")
    
    # Test different AUC-ROC calculations
    print("\n=== Testing Different AUC-ROC Calculations ===")
    
    # Method 1: Current implementation (WRONG)
    try:
        auc_wrong = roc_auc_score(y_true, y_proba[:, 0])
        print(f"1. Current method - roc_auc_score(y_true, y_proba[:, 0]): {auc_wrong:.4f}")
    except Exception as e:
        print(f"1. Current method failed: {e}")
    
    # Method 2: Correct for positive class 1
    try:
        auc_correct1 = roc_auc_score(y_true, y_proba[:, 1])
        print(f"2. Correct method 1 - roc_auc_score(y_true, y_proba[:, 1]): {auc_correct1:.4f}")
    except Exception as e:
        print(f"2. Method 1 failed: {e}")
    
    # Method 3: Multi-class approach
    try:
        auc_correct2 = roc_auc_score(y_true, y_proba, multi_class='ovr')
        print(f"3. Multi-class method - roc_auc_score(y_true, y_proba, multi_class='ovr'): {auc_correct2:.4f}")
    except Exception as e:
        print(f"3. Multi-class method failed: {e}")
    
    # Method 4: Flip labels to make cancer positive
    try:
        y_true_flipped = 1 - y_true  # 0->1, 1->0
        auc_correct3 = roc_auc_score(y_true_flipped, y_proba[:, 0])
        print(f"4. Flipped labels - roc_auc_score(1-y_true, y_proba[:, 0]): {auc_correct3:.4f}")
    except Exception as e:
        print(f"4. Flipped labels method failed: {e}")
    
    # Method 5: Use predictions to verify our understanding
    y_pred = np.argmax(y_proba, axis=1)
    
    # Verify confusion matrix matches what we expect
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n=== Verification ===")
    print("Expected confusion matrix:")
    print(f"Cancer:  [{cancer_correct:5d}] [{cancer_wrong:5d}]")
    print(f"Healthy: [{healthy_wrong:5d}] [{healthy_correct:5d}]")
    print("\nActual confusion matrix:")
    print(f"Cancer:  [{cm[0,0]:5d}] [{cm[0,1]:5d}]")
    print(f"Healthy: [{cm[1,0]:5d}] [{cm[1,1]:5d}]")
    
    # Calculate other metrics to verify
    from sklearn.metrics import recall_score, precision_score
    
    sensitivity = recall_score(y_true, y_pred, pos_label=0)
    specificity = recall_score(y_true, y_pred, pos_label=1) 
    
    print(f"\nMetrics verification:")
    print(f"Sensitivity (Cancer recall): {sensitivity:.4f}")
    print(f"Specificity (Healthy recall): {specificity:.4f}")
    
    # Show the fundamental issue
    print(f"\n=== The Core Issue ===")
    print("In your setup:")
    print("- y_true: 0=Cancer, 1=Healthy")
    print("- y_proba[:, 0]: Probability of Cancer") 
    print("- y_proba[:, 1]: Probability of Healthy")
    print()
    print("But roc_auc_score(y_true, scores) expects:")
    print("- y_true: 0=Negative class, 1=Positive class")
    print("- scores: Probability of POSITIVE class (1)")
    print()
    print("So when you pass roc_auc_score(y_true, y_proba[:, 0]):")
    print("- You're saying 'treat Healthy as positive class'")
    print("- But giving probabilities for Cancer")
    print("- This creates inverted probabilities!")
    
if __name__ == "__main__":
    debug_auc_roc_calculation()