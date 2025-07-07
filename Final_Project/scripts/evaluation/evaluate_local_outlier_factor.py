#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Local Outlier Factor Evaluation
------------------------------------------
This script evaluates the Local Outlier Factor model with optimal parameters.
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
from sklearn.neighbors import LocalOutlierFactor
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

# Calculate anomaly ratio for better contamination parameter
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

# Run Local Outlier Factor with optimized parameters
logger.info("Running Local Outlier Factor model with optimal parameters...")

# Start timing
start_time = time.time()

# Create and train LOF model
lof = LocalOutlierFactor(
    n_neighbors=50,  # Increased from default 20
    contamination=anomaly_ratio * 0.8,  # Slightly lower for better precision
    algorithm='auto',
    leaf_size=40,
    metric='euclidean',
    novelty=True,
    n_jobs=-1
)
lof.fit(X_train)

training_time = time.time() - start_time
logger.info(f"Training time: {training_time:.2f} seconds")

# Get predictions
start_time = time.time()
y_pred_raw = lof.predict(X_test)
prediction_time = time.time() - start_time
y_pred = np.array([1 if label == -1 else 0 for label in y_pred_raw])

logger.info(f"Prediction time: {prediction_time:.2f} seconds")

# Evaluate LOF
lof_metrics = evaluate_model("Local Outlier Factor", y_test, y_pred)

# Save LOF model
joblib.dump(lof, os.path.join(MODELS_DIR, 'local_outlier_factor_best.joblib'))

# Print detailed metrics
logger.info("\nDetailed Local Outlier Factor Metrics:")
logger.info(f"Accuracy: {lof_metrics['accuracy']:.4f}")
logger.info(f"Precision: {lof_metrics['precision']:.4f}")
logger.info(f"Recall: {lof_metrics['recall']:.4f}")
logger.info(f"F1-score: {lof_metrics['f1_score']:.4f}")
logger.info(f"Specificity: {lof_metrics['specificity']:.4f}")
logger.info(f"True Positives: {lof_metrics['true_positive']}")
logger.info(f"True Negatives: {lof_metrics['true_negative']}")
logger.info(f"False Positives: {lof_metrics['false_positive']}")
logger.info(f"False Negatives: {lof_metrics['false_negative']}")

logger.info("\nLocal Outlier Factor model evaluation completed successfully.") 