#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Best Model Evaluation (DBSCAN)
------------------------------------------
This script evaluates the DBSCAN model with optimal parameters.
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
from sklearn.cluster import DBSCAN

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
X_test = np.load(os.path.join(DATA_DIR, 'processed', 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'processed', 'y_test.npy'))

logger.info(f"Data loaded. X_test shape: {X_test.shape}")

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

# Run DBSCAN with optimal parameters
logger.info("Running DBSCAN model with optimal parameters...")
model_dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
cluster_labels = model_dbscan.fit_predict(X_test)

# Map clusters to labels
clusters = np.unique(cluster_labels)
cluster_to_label = {}

# For each cluster, assign the majority label
for cluster in clusters:
    if cluster == -1:  # Noise points are anomalies
        cluster_to_label[cluster] = 1
    else:
        mask = (cluster_labels == cluster)
        normal_count = np.sum(y_test[mask] == 0)
        anomaly_count = np.sum(y_test[mask] == 1)
        cluster_to_label[cluster] = 0 if normal_count > anomaly_count else 1

# Map clusters to binary labels
y_pred_dbscan = np.zeros_like(cluster_labels)
for i, cluster in enumerate(cluster_labels):
    y_pred_dbscan[i] = cluster_to_label.get(cluster, 1)  # Default to anomaly

# Evaluate DBSCAN
dbscan_metrics = evaluate_model("DBSCAN", y_test, y_pred_dbscan)

# Save DBSCAN model
dbscan_model = {
    'model': model_dbscan,
    'cluster_to_label': cluster_to_label,
    'eps': 0.5,
    'min_samples': 5
}
joblib.dump(dbscan_model, os.path.join(MODELS_DIR, 'dbscan_best.joblib'))

# Display cluster distribution
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
logger.info(f"Cluster distribution: {cluster_counts.to_dict()}")

# Print detailed metrics
logger.info("\nDetailed DBSCAN Metrics:")
logger.info(f"Accuracy: {dbscan_metrics['accuracy']:.4f}")
logger.info(f"Precision: {dbscan_metrics['precision']:.4f}")
logger.info(f"Recall: {dbscan_metrics['recall']:.4f}")
logger.info(f"F1-score: {dbscan_metrics['f1_score']:.4f}")
logger.info(f"Specificity: {dbscan_metrics['specificity']:.4f}")
logger.info(f"True Positives: {dbscan_metrics['true_positive']}")
logger.info(f"True Negatives: {dbscan_metrics['true_negative']}")
logger.info(f"False Positives: {dbscan_metrics['false_positive']}")
logger.info(f"False Negatives: {dbscan_metrics['false_negative']}")

logger.info("\nDBSCAN model evaluation completed successfully.") 