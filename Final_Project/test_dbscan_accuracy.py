#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify DBSCAN model accuracy
-------------------------------------------
This script loads the DBSCAN model and tests its accuracy on the test data.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN

# Define paths
DATA_DIR = "data"
MODELS_DIR = "models"

# Load data
print("Loading data...")
X_test = np.load(os.path.join(DATA_DIR, 'processed', 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'processed', 'y_test.npy'))

print(f"Data loaded. X_test shape: {X_test.shape}")

# Function to evaluate model
def evaluate_model(name, model, X_test, y_test):
    if name == "DBSCAN" and isinstance(model, dict) and 'model' in model:
        # Extract parameters from the model
        eps = model['model'].get_params()['eps']
        min_samples = model['model'].get_params()['min_samples']
        
        # Create a new DBSCAN instance with these parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        
        # Fit on the test data
        cluster_labels = dbscan.fit_predict(X_test)
        
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
        y_pred = np.zeros_like(cluster_labels)
        for i, cluster in enumerate(cluster_labels):
            y_pred[i] = cluster_to_label.get(cluster, 1)  # Default to anomaly
    
    elif name == "DBSCAN" and hasattr(model, 'fit_predict'):
        # Direct DBSCAN instance
        cluster_labels = model.fit_predict(X_test)
        
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
        y_pred = np.zeros_like(cluster_labels)
        for i, cluster in enumerate(cluster_labels):
            y_pred[i] = cluster_to_label.get(cluster, 1)  # Default to anomaly
    
    else:
        # For other models
        y_pred_raw = model.predict(X_test)
        y_pred = np.array([1 if label == -1 else 0 for label in y_pred_raw])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"{name} metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Test all available DBSCAN models
dbscan_models = [
    'dbscan_improved.joblib',
    'dbscan_best.joblib',
    'dbscan_fixed.joblib',
    'dbscan.joblib'
]

for model_file in dbscan_models:
    model_path = os.path.join(MODELS_DIR, model_file)
    if os.path.exists(model_path):
        print(f"\nTesting {model_file}...")
        try:
            model = joblib.load(model_path)
            evaluate_model("DBSCAN", model, X_test, y_test)
        except Exception as e:
            print(f"Error loading or evaluating {model_file}: {e}")
    else:
        print(f"Model file not found: {model_file}")

# Create a DBSCAN model with the parameters from run_improved_models.py
print("\nTesting DBSCAN with parameters from run_improved_models.py...")
model_dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
dbscan_model = {
    'model': model_dbscan,
    'eps': 0.5,
    'min_samples': 5
}
evaluate_model("DBSCAN", dbscan_model, X_test, y_test)

print("\nTest complete.") 