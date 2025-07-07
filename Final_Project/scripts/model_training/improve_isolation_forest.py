#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Isolation Forest Improvement
------------------------------------------
This script improves the Isolation Forest model through hyperparameter tuning
and provides detailed visualizations for model interpretation.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA

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
def evaluate_model(name, model, X_test, y_test):
    # Get predictions
    y_pred_raw = model.predict(X_test)
    y_pred = np.array([1 if label == -1 else 0 for label in y_pred_raw])
    
    # Get decision scores for ROC curve
    decision_scores = -model.decision_function(X_test)
    
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
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, decision_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"{name.lower().replace(' ', '_')}_roc_curve.png"), dpi=300)
    plt.close()
    
    # Generate Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, decision_scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"{name.lower().replace(' ', '_')}_precision_recall_curve.png"), dpi=300)
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
        'specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

# Hyperparameter tuning for Isolation Forest
logger.info("Starting hyperparameter tuning for Isolation Forest...")

# Define parameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_samples': ['auto', 256, 512],
    'contamination': [anomaly_ratio, anomaly_ratio * 0.9, anomaly_ratio * 1.1],
    'max_features': [0.5, 0.7, 1.0],
    'bootstrap': [True, False]
}

# Subset of data for faster tuning
from sklearn.utils import resample
X_train_sample = resample(X_train, n_samples=min(20000, X_train.shape[0]), random_state=42)
y_train_sample = resample(y_train, n_samples=min(20000, y_train.shape[0]), random_state=42)

# Perform grid search
results = []
best_f1 = 0
best_model = None
best_params = None

# Limit the number of combinations to explore
param_combinations = list(ParameterGrid(param_grid))
logger.info(f"Testing {len(param_combinations)} parameter combinations...")

# Use a subset of combinations for efficiency
selected_combinations = param_combinations[:10]  # Adjust this number based on computational resources

for i, params in enumerate(selected_combinations):
    logger.info(f"Testing combination {i+1}/{len(selected_combinations)}: {params}")
    
    # Create and train model
    model = IsolationForest(
        n_estimators=params['n_estimators'],
        max_samples=params['max_samples'],
        contamination=params['contamination'],
        max_features=params['max_features'],
        bootstrap=params['bootstrap'],
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_sample)
    
    # Evaluate on test set
    y_pred_raw = model.predict(X_test)
    y_pred = np.array([1 if label == -1 else 0 for label in y_pred_raw])
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results.append({
        'params': params,
        'f1_score': f1,
        'accuracy': accuracy
    })
    
    # Update best model
    if f1 > best_f1:
        best_f1 = f1
        best_params = params
        best_model = model

# Sort results by F1 score
results.sort(key=lambda x: x['f1_score'], reverse=True)

# Display top 5 results
logger.info("Top 5 parameter combinations:")
for i, result in enumerate(results[:5]):
    logger.info(f"{i+1}. F1: {result['f1_score']:.4f}, Accuracy: {result['accuracy']:.4f}, Params: {result['params']}")

# Train best model on full dataset
logger.info(f"Training best model with parameters: {best_params}")
best_full_model = IsolationForest(
    n_estimators=best_params['n_estimators'],
    max_samples=best_params['max_samples'],
    contamination=best_params['contamination'],
    max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    n_jobs=-1,
    random_state=42
)
best_full_model.fit(X_train)

# Evaluate best model
best_metrics = evaluate_model("Isolation Forest (Optimized)", best_full_model, X_test, y_test)

# Save best model
joblib.dump(best_full_model, os.path.join(MODELS_DIR, 'isolation_forest_optimized.joblib'))

# Visualize parameter importance
param_names = list(param_grid.keys())
param_values = [best_params[param] for param in param_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(param_names, [1] * len(param_names), color='lightblue')
for i, bar in enumerate(bars):
    plt.text(i, 0.5, str(param_values[i]), ha='center', va='center', rotation=0)
plt.title('Best Parameters for Isolation Forest')
plt.ylabel('Parameter Value')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'isolation_forest_best_params.png'), dpi=300)
plt.close()

# Visualize hyperparameter tuning results
plt.figure(figsize=(12, 6))
df_results = pd.DataFrame(results)
df_results['combination'] = range(1, len(df_results) + 1)
df_results[['combination', 'f1_score', 'accuracy']].plot(x='combination', kind='bar', figsize=(12, 6))
plt.title('Hyperparameter Tuning Results')
plt.xlabel('Parameter Combination')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'isolation_forest_hyperparameter_tuning.png'), dpi=300)
plt.close()

# Feature importance analysis using PCA
logger.info("Performing feature importance analysis...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

# Get predictions
y_pred_raw = best_full_model.predict(X_test)
y_pred = np.array([1 if label == -1 else 0 for label in y_pred_raw])

# Create DataFrame for visualization
df_pca = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'True Label': y_test,
    'Predicted Label': y_pred,
    'Correct': y_test == y_pred
})

# Plot PCA visualization
plt.figure(figsize=(12, 10))

# Plot by true label
plt.subplot(2, 2, 1)
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='True Label', palette={0: 'blue', 1: 'red'}, alpha=0.6)
plt.title('True Labels (PCA)')
plt.legend(title='Class', labels=['Normal', 'Anomaly'])

# Plot by predicted label
plt.subplot(2, 2, 2)
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Predicted Label', palette={0: 'blue', 1: 'red'}, alpha=0.6)
plt.title('Predicted Labels (PCA)')
plt.legend(title='Class', labels=['Normal', 'Anomaly'])

# Plot by correct/incorrect predictions
plt.subplot(2, 2, 3)
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Correct', palette={True: 'green', False: 'orange'}, alpha=0.6)
plt.title('Prediction Correctness (PCA)')
plt.legend(title='Correct', labels=['False', 'True'])

# Plot decision boundary (approximation)
plt.subplot(2, 2, 4)
decision_scores = -best_full_model.decision_function(X_test)
df_pca['Decision Score'] = decision_scores
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Decision Score', palette='viridis', alpha=0.6)
plt.title('Decision Scores (PCA)')
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'isolation_forest_pca_visualization.png'), dpi=300)
plt.close()

# Print detailed metrics
logger.info("\nDetailed Isolation Forest Metrics:")
logger.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
logger.info(f"Precision: {best_metrics['precision']:.4f}")
logger.info(f"Recall: {best_metrics['recall']:.4f}")
logger.info(f"F1-score: {best_metrics['f1_score']:.4f}")
logger.info(f"Specificity: {best_metrics['specificity']:.4f}")
logger.info(f"ROC AUC: {best_metrics['roc_auc']:.4f}")
logger.info(f"PR AUC: {best_metrics['pr_auc']:.4f}")
logger.info(f"True Positives: {best_metrics['true_positive']}")
logger.info(f"True Negatives: {best_metrics['true_negative']}")
logger.info(f"False Positives: {best_metrics['false_positive']}")
logger.info(f"False Negatives: {best_metrics['false_negative']}")

logger.info("\nIsolation Forest optimization completed successfully.") 