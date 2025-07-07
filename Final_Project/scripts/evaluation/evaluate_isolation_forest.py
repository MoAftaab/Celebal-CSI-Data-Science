#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Isolation Forest Evaluation
------------------------------------------
This script evaluates the Isolation Forest model with optimal parameters.
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
from sklearn.ensemble import IsolationForest

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

# Run Isolation Forest with improved parameters
logger.info("Running Isolation Forest model with optimal parameters...")
iso_forest = IsolationForest(
    n_estimators=500,  # Increased from default 100
    max_samples=min(512, X_train.shape[0]),  # Explicit sample size
    contamination=anomaly_ratio,  # Using actual anomaly ratio
    max_features=0.7,  # Feature subsampling
    bootstrap=True,  # Bootstrap sampling
    n_jobs=-1,  # Parallel processing
    random_state=42
)
iso_forest.fit(X_train)

# Get Isolation Forest predictions
y_pred_raw_iso = iso_forest.predict(X_test)
y_pred_iso = np.array([1 if label == -1 else 0 for label in y_pred_raw_iso])

# Evaluate Isolation Forest
iso_metrics = evaluate_model("Isolation Forest", y_test, y_pred_iso)

# Save Isolation Forest model
joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'isolation_forest_best.joblib'))

# Print detailed metrics
logger.info("\nDetailed Isolation Forest Metrics:")
logger.info(f"Accuracy: {iso_metrics['accuracy']:.4f}")
logger.info(f"Precision: {iso_metrics['precision']:.4f}")
logger.info(f"Recall: {iso_metrics['recall']:.4f}")
logger.info(f"F1-score: {iso_metrics['f1_score']:.4f}")
logger.info(f"Specificity: {iso_metrics['specificity']:.4f}")
logger.info(f"True Positives: {iso_metrics['true_positive']}")
logger.info(f"True Negatives: {iso_metrics['true_negative']}")
logger.info(f"False Positives: {iso_metrics['false_positive']}")
logger.info(f"False Negatives: {iso_metrics['false_negative']}")

logger.info("\nIsolation Forest model evaluation completed successfully.") 