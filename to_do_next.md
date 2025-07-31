


Add Early stopping as an onpage param
fix the way Training Mode Manual Training & Hyperparameter Optimization is displayed on the app, it currently breaks on the page, input can be prettified too
On the app in Optimization Status while using optuna, the title for metrics and params (Trial	Recall	Specificity	AUC-ROC	MCC	Parameters) are written in white on a whitish background so they currently aren't visible, can you fix this ?
Can you check if we are sure there are 279 Patients in our dataset ?

Ask if we can add a densenet model and attempt to make one


# may already be fixed

needs to check why when making a training with optuna it still displays the F1 and shouldn't it be replaced by monitored metrics amongst recall MCC or AUC-ROC, currrently we see this: 
2025-07-31 15:41:09,596 - INFO - M4 Pro optimizations enabled: Mixed Precision disabled on MPS (PyTorch limitation)
2025-07-31 15:41:09,596 - INFO - Starting training for 30 epochs
2025-07-31 15:41:09,596 - INFO - Monitoring metric: val_f1_score (val_mcc)
 Trial 11 completed. Best F1: 0.9220
but also on the app it shows that

# NEED NEW BRANCH

Possible for us to show the attention map on an actual image  with the VIT model for interpretation ? Making sure the area is readable




# DONE


ok VIT with optuna works properly but when running the resnet some trials don't go throught because of some hyper params errors that seems avoidable since other trials don't encounter issues.
[16:30:26] Trial 1 FAILED: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])...
[16:30:26] Trial 1 completed. Best F1: 0.0000

[16:30:27] Trial 3: Testing {'learning_rate': 7.014786718657618e-06, 'epochs': 12, 'architecture': 'lightweight', 'pretrained': False, 'fine_tune_layers': 0, 'batch_size': 256}