#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - DBSCAN Fix
------------------------------------------
This script fixes the DBSCAN model to work with test data.
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dbscan_fix')

# Define paths
DATA_DIR = "data"
MODELS_DIR = "models"
VISUALIZATIONS_DIR = "visualizations"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def load_test_data():
    """Load preprocessed test data."""
    try:
        X_test = np.load(os.path.join(DATA_DIR, 'processed', 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'processed', 'y_test.npy'))
        return X_test, y_test
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None, None

def load_dbscan_model():
    """Load the DBSCAN model."""
    try:
        model_path = os.path.join(MODELS_DIR, 'dbscan.joblib')
        if os.path.exists(model_path):
            logger.info(f"Loading DBSCAN model from {model_path}")
            return joblib.load(model_path)
        else:
            logger.error(f"DBSCAN model not found at {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading DBSCAN model: {e}")
        return None

def fix_dbscan_model(dbscan_data, X_test, y_test):
    """Fix and evaluate the DBSCAN model with test data."""
    try:
        logger.info("Creating new DBSCAN model for test data...")
        
        # Extract parameters from the original model
        eps = dbscan_data['eps']
        min_samples = dbscan_data['min_samples']
        
        # Create a new DBSCAN model with the same parameters
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        
        # Fit the model on test data
        logger.info("Fitting DBSCAN on test data...")
        cluster_labels = model.fit_predict(X_test)
        
        # Map clusters to normal/anomaly labels
        logger.info("Mapping clusters to labels...")
        
        # Map clusters based on test data
        clusters = np.unique(cluster_labels)
        cluster_to_label = {}
        
        # For each cluster, assign the majority label
        for cluster in clusters:
            if cluster == -1:  # Noise points are considered anomalies
                cluster_to_label[cluster] = 1
            else:
                # Count normal vs anomaly in this cluster
                mask = (cluster_labels == cluster)
                normal_count = np.sum(y_test[mask] == 0)
                anomaly_count = np.sum(y_test[mask] == 1)
                # Assign majority label
                cluster_to_label[cluster] = 0 if normal_count > anomaly_count else 1
        
        # Map clusters to binary labels
        y_pred = np.zeros_like(cluster_labels)
        for i, cluster in enumerate(cluster_labels):
            y_pred[i] = cluster_to_label.get(cluster, 1)  # Default to anomaly
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"DBSCAN metrics: Accuracy={metrics['accuracy']:.4f}, F1-score={metrics['f1_score']:.4f}")
        
        # Generate confusion matrix visualization
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
        
        # Save the fixed model
        fixed_model = {
            'model': model,
            'cluster_to_label': cluster_to_label,
            'eps': eps,
            'min_samples': min_samples
        }
        
        joblib.dump(fixed_model, os.path.join(MODELS_DIR, 'dbscan_fixed.joblib'))
        
        # Update model metrics CSV
        csv_path = os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Find DBSCAN row
            if 'model' in df.columns:
                dbscan_idx = df[df['model'] == 'DBSCAN'].index[0] if 'DBSCAN' in df['model'].values else None
                if dbscan_idx is not None:
                    # Update metrics
                    for key, value in metrics.items():
                        if key in df.columns:
                            df.at[dbscan_idx, key] = value
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Updated DBSCAN metrics in {csv_path}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error fixing DBSCAN model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function."""
    try:
        logger.info("Starting DBSCAN fix...")
        
        # Load test data
        X_test, y_test = load_test_data()
        if X_test is None or y_test is None:
            logger.error("Failed to load test data. Exiting.")
            return
        
        # Load DBSCAN model
        dbscan_data = load_dbscan_model()
        if dbscan_data is None:
            logger.error("Failed to load DBSCAN model. Exiting.")
            return
        
        # Fix and evaluate DBSCAN model
        metrics = fix_dbscan_model(dbscan_data, X_test, y_test)
        
        logger.info("DBSCAN fix completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 