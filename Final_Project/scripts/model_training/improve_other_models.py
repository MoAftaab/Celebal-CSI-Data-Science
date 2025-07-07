import os
import numpy as np
import joblib
import logging
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Calculate anomaly ratio for better contamination parameter
anomaly_ratio = np.sum(y_train == 1) / len(y_train)
logger.info(f"Anomaly ratio in training data: {anomaly_ratio:.4f}")

# Function to evaluate and save model results
def evaluate_model(model, name, X_test, y_test):
    # Get predictions
    y_pred_raw = model.predict(X_test)
    
    # Convert predictions to binary format (0: normal, 1: anomaly)
    if name in ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]:
        y_pred = np.array([1 if label == -1 else 0 for label in y_pred_raw])
    else:
        y_pred = y_pred_raw
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"{name} metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
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
    
    # Generate ROC curve if model has decision_function
    if hasattr(model, "decision_function"):
        from sklearn.metrics import roc_curve, auc
        # For anomaly detectors, we need to negate the scores
        y_score = -model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"{name.lower().replace(' ', '_')}_roc_curve.png"), dpi=300)
        plt.close()
    
    # Update model metrics CSV
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model': name
    }
    
    csv_path = os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if name in df['model'].values:
            idx = df[df['model'] == name].index[0]
            for key, value in metrics.items():
                if key in df.columns:
                    df.at[idx, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
        df.to_csv(csv_path, index=False)
    
    return metrics

# 1. Improve Isolation Forest
logger.info("Creating improved Isolation Forest model...")
iso_forest = IsolationForest(
    n_estimators=500,  # Increased from 300 to 500
    max_samples=min(512, X_train.shape[0]),  # Increased from 256
    contamination=anomaly_ratio,  # Use actual anomaly ratio
    max_features=0.7,  # Reduced from 0.8 for better generalization
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
iso_forest.fit(X_train)
joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'isolation_forest_improved.joblib'))
evaluate_model(iso_forest, "Isolation Forest", X_test, y_test)

# 2. Improve One-Class SVM
logger.info("Creating improved One-Class SVM model...")
# Use a subset of data for One-Class SVM due to computational complexity
if X_train.shape[0] > 10000:
    from sklearn.utils import resample
    X_train_svm = resample(X_train, n_samples=10000, random_state=42)
else:
    X_train_svm = X_train

one_class_svm = OneClassSVM(
    nu=anomaly_ratio * 0.9,  # Slightly lower than anomaly ratio for better precision
    kernel='rbf',
    gamma='scale',
    shrinking=True,
    cache_size=2000
)
one_class_svm.fit(X_train_svm)
joblib.dump(one_class_svm, os.path.join(MODELS_DIR, 'one_class_svm_improved.joblib'))
evaluate_model(one_class_svm, "One-Class SVM", X_test, y_test)

# 3. Improve Local Outlier Factor
logger.info("Creating improved Local Outlier Factor model...")
lof = LocalOutlierFactor(
    n_neighbors=50,  # Increased from 30 to 50
    contamination=anomaly_ratio * 0.8,  # Slightly lower for better precision
    algorithm='auto',
    leaf_size=40,  # Increased from default 30
    metric='euclidean',
    novelty=True,
    n_jobs=-1
)
lof.fit(X_train)
joblib.dump(lof, os.path.join(MODELS_DIR, 'local_outlier_factor_improved.joblib'))
evaluate_model(lof, "Local Outlier Factor", X_test, y_test)

# Generate updated model comparison visualization
logger.info("Generating updated model comparison visualization...")
csv_path = os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    df.set_index('model')[metrics_to_plot].plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'improved_model_comparison.png'), dpi=300)
    plt.close()
    
    logger.info("Model comparison visualization saved.")

logger.info("Model improvements completed successfully.") 