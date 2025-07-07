#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Find Best Improved Model
------------------------------------------
This script evaluates all improved models and determines which has the best metrics.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(script_dir))

# Add the project root to the Python path
sys.path.append(project_root)

# Define paths
DATA_DIR = os.path.join(project_root, "data")
MODELS_DIR = os.path.join(project_root, "models")
VISUALIZATIONS_DIR = os.path.join(project_root, "visualizations")

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
    
    logger.info(f"{name} metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Return metrics
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

def main():
    # Load data
    logger.info("Loading test data...")
    try:
        X_test = np.load(os.path.join(DATA_DIR, 'processed', 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'processed', 'y_test.npy'))
        logger.info(f"Data loaded. X_test shape: {X_test.shape}")
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return

    # List of improved models to evaluate
    improved_models = [
        ('Isolation Forest (Improved)', 'isolation_forest_improved.joblib'),
        ('One-Class SVM (Improved)', 'one_class_svm_improved.joblib'),
        ('Local Outlier Factor (Improved)', 'local_outlier_factor_improved.joblib'),
        ('DBSCAN (Improved)', 'dbscan_improved.joblib')
    ]

    all_metrics = []

    # Evaluate each model
    for model_name, model_file in improved_models:
        model_path = os.path.join(MODELS_DIR, model_file)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            continue
            
        logger.info(f"Evaluating {model_name}...")
        
        # Load the model
        model = joblib.load(model_path)
        
        # Get predictions based on model type
        if 'dbscan' in model_file.lower():
            if isinstance(model, dict) and 'model' in model:
                # For DBSCAN, apply it directly to test data
                from sklearn.cluster import DBSCAN
                try:
                    eps = model['model'].get_params()['eps']
                    min_samples = model['model'].get_params()['min_samples']
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    dbscan.fit(X_test)
                    # -1 is outlier (anomaly), convert to 1 for consistency
                    y_pred = np.array([1 if label == -1 else 0 for label in dbscan.labels_])
                except Exception as e:
                    logger.error(f"Error using DBSCAN model: {e}")
                    continue
            else:
                logger.error("DBSCAN model format not recognized")
                continue
        else:
            # For Isolation Forest, One-Class SVM, LOF
            y_pred_raw = model.predict(X_test)
            y_pred = np.array([1 if label == -1 else 0 for label in y_pred_raw])
        
        # Evaluate the model
        metrics = evaluate_model(model_name, y_test, y_pred)
        all_metrics.append(metrics)

    # Create metrics DataFrame
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Find best model for each metric
        best_accuracy_model = metrics_df.loc[metrics_df['accuracy'].idxmax()]
        best_f1_model = metrics_df.loc[metrics_df['f1_score'].idxmax()]
        best_precision_model = metrics_df.loc[metrics_df['precision'].idxmax()]
        best_recall_model = metrics_df.loc[metrics_df['recall'].idxmax()]
        
        # Print results
        print("\n" + "="*50)
        print("BEST MODELS BY METRICS")
        print("="*50)
        
        print(f"\nBest model by accuracy: {best_accuracy_model['model']}")
        print(f"  Accuracy: {best_accuracy_model['accuracy']:.4f}")
        print(f"  F1 Score: {best_accuracy_model['f1_score']:.4f}")
        print(f"  Precision: {best_accuracy_model['precision']:.4f}")
        print(f"  Recall: {best_accuracy_model['recall']:.4f}")
        
        print(f"\nBest model by F1 score: {best_f1_model['model']}")
        print(f"  F1 Score: {best_f1_model['f1_score']:.4f}")
        print(f"  Accuracy: {best_f1_model['accuracy']:.4f}")
        print(f"  Precision: {best_f1_model['precision']:.4f}")
        print(f"  Recall: {best_f1_model['recall']:.4f}")
        
        print(f"\nBest model by precision: {best_precision_model['model']}")
        print(f"  Precision: {best_precision_model['precision']:.4f}")
        print(f"  Accuracy: {best_precision_model['accuracy']:.4f}")
        print(f"  F1 Score: {best_precision_model['f1_score']:.4f}")
        print(f"  Recall: {best_precision_model['recall']:.4f}")
        
        print(f"\nBest model by recall: {best_recall_model['model']}")
        print(f"  Recall: {best_recall_model['recall']:.4f}")
        print(f"  Accuracy: {best_recall_model['accuracy']:.4f}")
        print(f"  F1 Score: {best_recall_model['f1_score']:.4f}")
        print(f"  Precision: {best_recall_model['precision']:.4f}")
        
        # Save metrics to CSV for reference
        metrics_df.to_csv(os.path.join(VISUALIZATIONS_DIR, 'improved_model_metrics_summary.csv'), index=False)
        logger.info(f"Metrics saved to {os.path.join(VISUALIZATIONS_DIR, 'improved_model_metrics_summary.csv')}")
        
        # Return best model names
        return {
            'best_accuracy': best_accuracy_model['model'],
            'best_f1': best_f1_model['model'],
            'best_precision': best_precision_model['model'],
            'best_recall': best_recall_model['model']
        }
    else:
        logger.error("No models were successfully evaluated")
        return None

if __name__ == "__main__":
    main() 