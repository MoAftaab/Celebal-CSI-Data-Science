#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - All Models Comparison
------------------------------------------
This script runs all models and compares their results.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
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

# Calculate anomaly ratio
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

# Run DBSCAN
logger.info("Running DBSCAN model...")
start_time = time.time()
dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
cluster_labels = dbscan.fit_predict(X_test)

# Map clusters to normal/anomaly labels
# Cluster -1 is considered as anomaly by default
# For other clusters, assign the majority class
unique_clusters = np.unique(cluster_labels)
cluster_to_label = {}

for cluster in unique_clusters:
    if cluster == -1:  # Noise points are considered anomalies
        cluster_to_label[cluster] = 1
    else:
        # Get indices of points in this cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        # Get true labels of these points
        true_labels = y_test[cluster_indices]
        # Assign the majority class
        if np.mean(true_labels) >= 0.5:  # If majority is anomaly (1)
            cluster_to_label[cluster] = 1
        else:
            cluster_to_label[cluster] = 0

# Map cluster labels to binary labels
dbscan_pred = np.array([cluster_to_label[label] for label in cluster_labels])
dbscan_time = time.time() - start_time
logger.info(f"DBSCAN completed in {dbscan_time:.2f} seconds")

# Run Isolation Forest
logger.info("Running Isolation Forest model...")
start_time = time.time()
isolation_forest = IsolationForest(
    n_estimators=500,
    max_samples=512,
    contamination=anomaly_ratio,
    max_features=0.7,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
isolation_forest.fit(X_train)
if_pred_raw = isolation_forest.predict(X_test)
if_pred = np.array([1 if label == -1 else 0 for label in if_pred_raw])
if_time = time.time() - start_time
logger.info(f"Isolation Forest completed in {if_time:.2f} seconds")

# Run One-Class SVM on a subset due to computational complexity
logger.info("Running One-Class SVM model...")
start_time = time.time()
X_train_subset = resample(X_train, n_samples=min(10000, X_train.shape[0]), random_state=42)
one_class_svm = OneClassSVM(
    nu=anomaly_ratio * 0.8,
    kernel='rbf',
    gamma='scale',
    shrinking=True,
    cache_size=2000
)
one_class_svm.fit(X_train_subset)
svm_pred_raw = one_class_svm.predict(X_test)
svm_pred = np.array([1 if label == -1 else 0 for label in svm_pred_raw])
svm_time = time.time() - start_time
logger.info(f"One-Class SVM completed in {svm_time:.2f} seconds")

# Run Local Outlier Factor
logger.info("Running Local Outlier Factor model...")
start_time = time.time()
lof = LocalOutlierFactor(
    n_neighbors=50,
    contamination=anomaly_ratio * 0.8,
    algorithm='auto',
    leaf_size=40,
    metric='euclidean',
    novelty=True,
    n_jobs=-1
)
lof.fit(X_train)
lof_pred_raw = lof.predict(X_test)
lof_pred = np.array([1 if label == -1 else 0 for label in lof_pred_raw])
lof_time = time.time() - start_time
logger.info(f"Local Outlier Factor completed in {lof_time:.2f} seconds")

# Evaluate all models
logger.info("\nEvaluating all models...")
dbscan_metrics = evaluate_model("DBSCAN", y_test, dbscan_pred)
if_metrics = evaluate_model("Isolation Forest", y_test, if_pred)
svm_metrics = evaluate_model("One-Class SVM", y_test, svm_pred)
lof_metrics = evaluate_model("Local Outlier Factor", y_test, lof_pred)

# Create a DataFrame with all metrics for comparison
metrics_df = pd.DataFrame([
    dbscan_metrics,
    if_metrics,
    svm_metrics,
    lof_metrics
])

# Display comparison table
logger.info("\nModel Comparison:")
logger.info(f"\n{metrics_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']]}")

# Save metrics to CSV
metrics_df.to_csv(os.path.join(VISUALIZATIONS_DIR, 'model_metrics_comparison.csv'), index=False)

# Create bar chart comparison
plt.figure(figsize=(12, 8))
metrics_df.set_index('model')[['accuracy', 'precision', 'recall', 'f1_score', 'specificity']].plot(kind='bar', figsize=(12, 8))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison_bar.png'), dpi=300)

# Create radar chart for visualization
plt.figure(figsize=(10, 8))
categories = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']
N = len(categories)

# Create angles for each metric
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Create radar plot
ax = plt.subplot(111, polar=True)

# Add lines and points for each model
colors = ['b', 'g', 'r', 'c']
for i, model in enumerate(metrics_df['model']):
    values = metrics_df.loc[i, ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']].values.flatten().tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

# Set category labels
plt.xticks(angles[:-1], categories)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Model Performance Comparison (Radar Chart)')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison_radar.png'), dpi=300)

logger.info("\nAll models comparison completed successfully.")
logger.info(f"Results saved to {os.path.join(VISUALIZATIONS_DIR, 'model_metrics_comparison.csv')}")
logger.info(f"Visualizations saved to {VISUALIZATIONS_DIR}")

# Determine the best model
best_model_idx = metrics_df['f1_score'].idxmax()
best_model = metrics_df.loc[best_model_idx, 'model']
logger.info(f"\nBest performing model based on F1-score: {best_model}")
logger.info(f"Best model metrics: Accuracy={metrics_df.loc[best_model_idx, 'accuracy']:.4f}, " +
            f"Precision={metrics_df.loc[best_model_idx, 'precision']:.4f}, " +
            f"Recall={metrics_df.loc[best_model_idx, 'recall']:.4f}, " +
            f"F1-score={metrics_df.loc[best_model_idx, 'f1_score']:.4f}, " +
            f"Specificity={metrics_df.loc[best_model_idx, 'specificity']:.4f}") 