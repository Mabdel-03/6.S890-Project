"""Evaluation metrics"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_accuracy(predictions, targets):
    """
    Compute accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Accuracy score
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
        
    return accuracy_score(targets, predictions)


def compute_metrics(predictions, targets):
    """
    Compute comprehensive metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
        
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='macro'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


