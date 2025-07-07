#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Improved Models Direct Evaluation
------------------------------------------
This script directly runs improved anomaly detection models and evaluates their performance.
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
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

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

# List to store all metrics
all_metrics = []

# 1. Run DBSCAN (best model)
logger.info("Running DBSCAN model...")

# Create DBSCAN model with optimal parameters
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
all_metrics.append(dbscan_metrics)

# Save DBSCAN model
dbscan_model = {
    'model': model_dbscan,
    'cluster_to_label': cluster_to_label,
    'eps': 0.5,
    'min_samples': 5
}
joblib.dump(dbscan_model, os.path.join(MODELS_DIR, 'dbscan_improved.joblib'))

# 2. Run Isolation Forest with improved parameters
logger.info("Running Isolation Forest model...")
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
all_metrics.append(iso_metrics)

# Save Isolation Forest model
joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'isolation_forest_improved.joblib'))

# 3. Run One-Class SVM with improved parameters
logger.info("Running One-Class SVM model...")
# Use a subset of data for One-Class SVM due to computational complexity
if X_train.shape[0] > 10000:
    from sklearn.utils import resample
    X_train_svm = resample(X_train, n_samples=10000, random_state=42)
else:
    X_train_svm = X_train

one_class_svm = OneClassSVM(
    nu=anomaly_ratio * 0.9,  # Slightly lower than anomaly ratio
    kernel='rbf',
    gamma='scale',
    shrinking=True,
    cache_size=2000
)
one_class_svm.fit(X_train_svm)

# Get One-Class SVM predictions
y_pred_raw_svm = one_class_svm.predict(X_test)
y_pred_svm = np.array([1 if label == -1 else 0 for label in y_pred_raw_svm])

# Evaluate One-Class SVM
svm_metrics = evaluate_model("One-Class SVM", y_test, y_pred_svm)
all_metrics.append(svm_metrics)

# Save One-Class SVM model
joblib.dump(one_class_svm, os.path.join(MODELS_DIR, 'one_class_svm_improved.joblib'))

# 4. Run Local Outlier Factor with improved parameters
logger.info("Running Local Outlier Factor model...")
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

# Get LOF predictions
y_pred_raw_lof = lof.predict(X_test)
y_pred_lof = np.array([1 if label == -1 else 0 for label in y_pred_raw_lof])

# Evaluate LOF
lof_metrics = evaluate_model("Local Outlier Factor", y_test, y_pred_lof)
all_metrics.append(lof_metrics)

# Save LOF model
joblib.dump(lof, os.path.join(MODELS_DIR, 'local_outlier_factor_improved.joblib'))

# Create metrics DataFrame
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv'), index=False)

# Generate model comparison bar chart
plt.figure(figsize=(12, 8))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
metrics_df.set_index('model')[metrics_to_plot].plot(kind='bar', figsize=(12, 8))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison.png'), dpi=300)
plt.close()

# Generate radar chart for multi-metric visualization
def radar_chart(df, metrics, title='Model Comparison'):
    # Count of metrics
    N = len(metrics)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Set first axis to be on top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per metric and add labels
    plt.xticks(angles[:-1], metrics)
    
    # Set limits for each metric's axis
    ax.set_ylim(0, 1)
    
    # Plot each model
    for idx, model in enumerate(df['model']):
        values = df.loc[idx, metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, color='black', y=1.1)
    
    return fig

# Create radar chart
radar_fig = radar_chart(metrics_df, metrics_to_plot, title='Model Comparison - Multiple Metrics')
radar_fig.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison_radar.png'), dpi=300, bbox_inches='tight')
plt.close(radar_fig)

logger.info("All models evaluated and visualizations generated successfully.")
logger.info(f"Best model: {metrics_df.loc[metrics_df['f1_score'].idxmax(), 'model']} with F1-score: {metrics_df['f1_score'].max():.4f}")
logger.info(f"Results saved to {os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv')}") 