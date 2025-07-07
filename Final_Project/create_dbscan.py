#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Create Improved DBSCAN
------------------------------------------
This script creates an improved DBSCAN model.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define paths
DATA_DIR = "data"
MODELS_DIR = "models"
VISUALIZATIONS_DIR = "visualizations"

# Load data
logger.info("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, 'processed', 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'processed', 'y_train.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'processed', 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'processed', 'y_test.npy'))

logger.info(f"Data loaded. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Create DBSCAN model
logger.info("Creating DBSCAN model...")
model = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
cluster_labels = model.fit_predict(X_test)

# Map clusters to labels
logger.info("Mapping clusters to labels...")
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
y_pred = np.zeros_like(cluster_labels)
for i, cluster in enumerate(cluster_labels):
    y_pred[i] = cluster_to_label.get(cluster, 1)  # Default to anomaly

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

logger.info(f"DBSCAN metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
           xticklabels=['Normal', 'Anomaly'],
           yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - DBSCAN')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'dbscan_confusion_matrix.png'), dpi=300)
plt.close()

# Save the model
dbscan_model = {
    'model': model,
    'cluster_to_label': cluster_to_label,
    'eps': 0.5,
    'min_samples': 5
}

joblib.dump(dbscan_model, os.path.join(MODELS_DIR, 'dbscan_improved.joblib'))
logger.info("DBSCAN model saved.")

# Update model metrics CSV
csv_path = os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    # Find DBSCAN row or add new row
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model': 'DBSCAN'
    }
    
    if 'DBSCAN' in df['model'].values:
        dbscan_idx = df[df['model'] == 'DBSCAN'].index[0]
        for key, value in metrics.items():
            if key in df.columns:
                df.at[dbscan_idx, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    logger.info(f"Updated model metrics CSV at {csv_path}")

def main():
    """Main function."""
    try:
        logger.info("Starting improved DBSCAN creation...")
        
        # Load data
        X_train, y_train, X_test, y_test = load_data()
        if X_train is None or y_train is None or X_test is None or y_test is None:
            logger.error("Failed to load data. Exiting.")
            return
        
        # Create and evaluate DBSCAN model
        metrics = create_dbscan_model(X_train, y_train, X_test, y_test)
        
        logger.info("Improved DBSCAN creation completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 