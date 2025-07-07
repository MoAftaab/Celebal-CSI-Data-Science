#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Model Utilities
-----------------------------------------
This module contains utility functions for model operations and evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    auc, accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_utils')

# Try to import tensorflow
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available. Some features may be limited.")


def load_model_from_file(model_path):
    """
    Load a trained model from a file.
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
    
    Returns:
    --------
    object
        The loaded model
    """
    try:
        # Check file extension
        if model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        elif model_path.endswith('.h5'):
            if HAS_TENSORFLOW:
                model = load_model(model_path)
            else:
                logger.error("TensorFlow not available. Cannot load .h5 model files.")
                return None
        else:
            logger.error(f"Unsupported model file format: {model_path}")
            return None
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def save_model(model, model_path):
    """
    Save a trained model to a file.
    
    Parameters:
    -----------
    model : object
        The model to save
    model_path : str
        Path where to save the model
    
    Returns:
    --------
    bool
        True if the model was saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Check model type and file extension
        if model_path.endswith('.joblib'):
            joblib.dump(model, model_path)
        elif model_path.endswith('.h5'):
            if not HAS_TENSORFLOW:
                logger.error("TensorFlow not available. Cannot save model in .h5 format.")
                return False
                
            if hasattr(model, 'save'):
                model.save(model_path)
            else:
                logger.error("Model does not have a 'save' method for .h5 format")
                return False
        else:
            logger.error(f"Unsupported model file format: {model_path}")
            return False
        
        logger.info(f"Model saved successfully to {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def get_anomaly_scores(model, X):
    """
    Get anomaly scores from a model.
    
    Parameters:
    -----------
    model : object
        The trained model
    X : numpy.ndarray
        Data to get scores for
    
    Returns:
    --------
    numpy.ndarray
        Anomaly scores (higher = more anomalous)
    """
    # Try different methods to get anomaly scores
    try:
        if hasattr(model, 'get_anomaly_scores'):
            # Custom method (from our model classes)
            return model.get_anomaly_scores(X)
        elif hasattr(model, 'decision_function'):
            # For models like OneClassSVM, Isolation Forest
            # Lower scores typically indicate anomalies, so negate them
            return -model.decision_function(X)
        elif hasattr(model, 'score_samples'):
            # For models like LocalOutlierFactor in novelty mode
            # Lower scores typically indicate anomalies, so negate them
            return -model.score_samples(X)
        elif hasattr(model, 'predict_proba'):
            # For models with probability output
            proba = model.predict_proba(X)
            if proba.shape[1] >= 2:
                # Return probability of the anomaly class (typically class 1)
                return proba[:, 1]
            else:
                return proba
        elif hasattr(model, 'predict'):
            # For Keras/TF models (reconstruction error)
            if hasattr(model, 'predict'):
                reconstructions = model.predict(X)
                mse = np.mean(np.power(X - reconstructions, 2), axis=1)
                return mse
        else:
            logger.error(f"No method found to get anomaly scores from model {type(model).__name__}")
            return np.zeros(X.shape[0])
    
    except Exception as e:
        logger.error(f"Error getting anomaly scores: {e}")
        return np.zeros(X.shape[0])


def evaluate_binary_classification(y_true, y_pred, y_score=None):
    """
    Evaluate binary classification performance.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (0: normal, 1: anomaly)
    y_pred : numpy.ndarray
        Predicted labels (0: normal, 1: anomaly)
    y_score : numpy.ndarray or None
        Predicted scores for computing ROC and PR curves
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    try:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Initialize results dictionary
        results = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
        
        # Compute ROC and PR curves if scores are provided
        if y_score is not None:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # PR curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall_curve, precision_curve)
            
            # Add to results
            results.update({
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fpr': fpr,
                'tpr': tpr,
                'precision_curve': precision_curve,
                'recall_curve': recall_curve
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating binary classification: {e}")
        return None


def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6), cmap='Blues', title='Confusion Matrix'):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    class_names : list or None
        Names of the classes
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the heatmap
    title : str
        Title of the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Set default class names if not provided
        if class_names is None:
            class_names = ['Normal', 'Anomaly']
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap=cmap, 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        return None


def plot_roc_curve(fpr, tpr, roc_auc, figsize=(8, 6), title='Receiver Operating Characteristic (ROC) Curve'):
    """
    Plot a ROC curve.
    
    Parameters:
    -----------
    fpr : numpy.ndarray
        False positive rates
    tpr : numpy.ndarray
        True positive rates
    roc_auc : float
        Area under the ROC curve
    figsize : tuple
        Figure size
    title : str
        Title of the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        
        # Plot random guessing line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")
        return None


def plot_precision_recall_curve(precision, recall, pr_auc, figsize=(8, 6), title='Precision-Recall Curve'):
    """
    Plot a Precision-Recall curve.
    
    Parameters:
    -----------
    precision : numpy.ndarray
        Precision values
    recall : numpy.ndarray
        Recall values
    pr_auc : float
        Area under the Precision-Recall curve
    figsize : tuple
        Figure size
    title : str
        Title of the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot PR curve
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting Precision-Recall curve: {e}")
        return None


def find_optimal_threshold(y_true, y_score, metric='f1'):
    """
    Find the optimal threshold for binary classification based on a metric.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (0: normal, 1: anomaly)
    y_score : numpy.ndarray
        Predicted scores
    metric : str
        Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
    
    Returns:
    --------
    float
        Optimal threshold
    """
    try:
        # Get sorted unique scores for thresholds
        thresholds = np.unique(y_score)
        
        # If there are too many thresholds, sample them
        if len(thresholds) > 100:
            thresholds = np.percentile(y_score, np.linspace(0, 100, 100))
        
        best_metric = -np.inf
        best_threshold = 0
        
        # Evaluate each threshold
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            else:
                logger.error(f"Unsupported metric: {metric}")
                return None
            
            if score > best_metric:
                best_metric = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold found: {best_threshold} with {metric} = {best_metric:.4f}")
        return best_threshold
    
    except Exception as e:
        logger.error(f"Error finding optimal threshold: {e}")
        return None


def plot_threshold_metrics(y_true, y_score, figsize=(10, 6)):
    """
    Plot metrics at different thresholds.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (0: normal, 1: anomaly)
    y_score : numpy.ndarray
        Predicted scores
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Get sorted unique scores for thresholds
        thresholds = np.unique(y_score)
        
        # If there are too many thresholds, sample them
        if len(thresholds) > 100:
            thresholds = np.percentile(y_score, np.linspace(0, 100, 100))
        
        # Initialize metric arrays
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # Evaluate each threshold
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            
            accuracies.append(accuracy_score(y_true, y_pred))
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot metrics
        plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
        plt.plot(thresholds, precisions, label='Precision', color='orange')
        plt.plot(thresholds, recalls, label='Recall', color='green')
        plt.plot(thresholds, f1_scores, label='F1 Score', color='red')
        
        # Find optimal thresholds for each metric
        optimal_accuracy_threshold = thresholds[np.argmax(accuracies)]
        optimal_precision_threshold = thresholds[np.argmax(precisions)]
        optimal_recall_threshold = thresholds[np.argmax(recalls)]
        optimal_f1_threshold = thresholds[np.argmax(f1_scores)]
        
        # Mark optimal thresholds
        plt.axvline(x=optimal_accuracy_threshold, color='blue', linestyle='--', alpha=0.5)
        plt.axvline(x=optimal_precision_threshold, color='orange', linestyle='--', alpha=0.5)
        plt.axvline(x=optimal_recall_threshold, color='green', linestyle='--', alpha=0.5)
        plt.axvline(x=optimal_f1_threshold, color='red', linestyle='--', alpha=0.5)
        
        # Set plot properties
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics at Different Thresholds')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting threshold metrics: {e}")
        return None


def ensemble_predictions(models, X, weights=None, threshold=0.5):
    """
    Generate ensemble predictions from multiple models.
    
    Parameters:
    -----------
    models : list
        List of trained models
    X : numpy.ndarray
        Input data
    weights : numpy.ndarray or None
        Weights for each model (if None, equal weights are used)
    threshold : float
        Threshold for binary classification
    
    Returns:
    --------
    tuple
        (binary_predictions, ensemble_scores)
    """
    try:
        # Get anomaly scores from each model
        all_scores = []
        
        for model in models:
            scores = get_anomaly_scores(model, X)
            
            # Normalize scores to [0, 1] range
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
            
            all_scores.append(normalized_scores)
        
        # Stack scores
        all_scores = np.column_stack(all_scores)
        
        # Apply weights if provided
        if weights is None:
            weights = np.ones(len(models)) / len(models)
        
        # Compute weighted average
        ensemble_scores = np.average(all_scores, axis=1, weights=weights)
        
        # Generate binary predictions
        binary_predictions = (ensemble_scores >= threshold).astype(int)
        
        return binary_predictions, ensemble_scores
    
    except Exception as e:
        logger.error(f"Error generating ensemble predictions: {e}")
        return None, None


def calibrate_ensemble_weights(models, X_val, y_val):
    """
    Calibrate weights for ensemble prediction using validation data.
    
    Parameters:
    -----------
    models : list
        List of trained models
    X_val : numpy.ndarray
        Validation data
    y_val : numpy.ndarray
        Validation labels (0: normal, 1: anomaly)
    
    Returns:
    --------
    numpy.ndarray
        Calibrated weights for each model
    """
    try:
        # Get anomaly scores from each model
        all_scores = []
        all_metrics = []
        
        for model in models:
            scores = get_anomaly_scores(model, X_val)
            
            # Normalize scores to [0, 1] range
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
            
            all_scores.append(normalized_scores)
            
            # Find optimal threshold
            threshold = find_optimal_threshold(y_val, normalized_scores, metric='f1')
            
            # Compute F1 score with optimal threshold
            y_pred = (normalized_scores >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            all_metrics.append(f1)
        
        # Convert metrics to weights
        # Higher F1 score = higher weight
        weights = np.array(all_metrics)
        
        # Ensure weights sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all metrics are 0, use equal weights
            weights = np.ones(len(models)) / len(models)
        
        logger.info(f"Calibrated ensemble weights: {weights}")
        return weights
    
    except Exception as e:
        logger.error(f"Error calibrating ensemble weights: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    pass 