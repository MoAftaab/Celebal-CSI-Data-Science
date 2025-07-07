#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - One-Class SVM Evaluation
------------------------------------------
This script evaluates the One-Class SVM model with optimal parameters.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.svm import OneClassSVM
from sklearn.utils import resample
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Define paths
DATA_DIR = "data"
MODELS_DIR = "models"
VISUALIZATIONS_DIR = "visualizations"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Load data
logger.info("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, 'processed', 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'processed', 'y_train.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'processed', 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'processed', 'y_test.npy'))

logger.info(f"Data loaded. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Calculate anomaly ratio for better nu parameter
anomaly_ratio = np.sum(y_train == 1) / len(y_train)
logger.info(f"Anomaly ratio in training data: {anomaly_ratio:.4f}")

# Function to evaluate model
def evaluate_model(name, y_test, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    logger.info(f"{name} metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Specificity={specificity:.4f}")
    
    # Generate confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"{name.lower().replace(' ', '_')}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Store metrics
    return {
        'model': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
        'specificity': specificity
    }

# Run One-Class SVM with optimized parameters
logger.info("Running One-Class SVM model with optimal parameters...")

# Use a subset of data for One-Class SVM due to computational complexity
logger.info("Creating subset of data for training...")
X_train_subset = resample(X_train, n_samples=min(10000, X_train.shape[0]), random_state=42)
logger.info(f"Training subset shape: {X_train_subset.shape}")

# Start timing
start_time = time.time()

# Create and train One-Class SVM model
one_class_svm = OneClassSVM(
    nu=anomaly_ratio * 0.8,  # Slightly lower than anomaly ratio
    kernel='rbf',
    gamma='scale',
    shrinking=True,
    cache_size=2000
)
one_class_svm.fit(X_train_subset)

training_time = time.time() - start_time
logger.info(f"Training time: {training_time:.2f} seconds")

# Get predictions
start_time = time.time()
y_pred_raw = one_class_svm.predict(X_test)
prediction_time = time.time() - start_time
y_pred = np.array([1 if label == -1 else 0 for label in y_pred_raw])

logger.info(f"Prediction time: {prediction_time:.2f} seconds")

# Evaluate One-Class SVM
svm_metrics = evaluate_model("One-Class SVM", y_test, y_pred)

# Save One-Class SVM model
joblib.dump(one_class_svm, os.path.join(MODELS_DIR, 'one_class_svm_best.joblib'))

# Print detailed metrics
logger.info("\nDetailed One-Class SVM Metrics:")
logger.info(f"Accuracy: {svm_metrics['accuracy']:.4f}")
logger.info(f"Precision: {svm_metrics['precision']:.4f}")
logger.info(f"Recall: {svm_metrics['recall']:.4f}")
logger.info(f"F1-score: {svm_metrics['f1_score']:.4f}")
logger.info(f"Specificity: {svm_metrics['specificity']:.4f}")
logger.info(f"True Positives: {svm_metrics['true_positive']}")
logger.info(f"True Negatives: {svm_metrics['true_negative']}")
logger.info(f"False Positives: {svm_metrics['false_positive']}")
logger.info(f"False Negatives: {svm_metrics['false_negative']}")

logger.info("\nOne-Class SVM model evaluation completed successfully.") 