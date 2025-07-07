#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Model Evaluation
------------------------------------------
This script evaluates the performance of anomaly detection models and generates visualizations.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_evaluation')

# Define paths
DATA_DIR = "data"
MODELS_DIR = "models"
VISUALIZATIONS_DIR = "visualizations"

# Create directories if they don't exist
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def load_dataset():
    """Load and preprocess the NSL-KDD dataset."""
    try:
        # Use full KDDTrain+ dataset instead of the small sample
        train_file_path = os.path.join(DATA_DIR, "datasets", "KDDTrain+.csv")
        test_file_path = os.path.join(DATA_DIR, "datasets", "KDDTest+.csv")
        
        # Check if files exist
        if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
            logger.error("Dataset files not found. Please run download_dataset.py first.")
            return None, None, None, None
        
        # Load the data
        logger.info(f"Loading train dataset from {train_file_path}")
        train_data = pd.read_csv(train_file_path, header=None)
        
        logger.info(f"Loading test dataset from {test_file_path}")
        test_data = pd.read_csv(test_file_path, header=None)
        
        # Preprocess the data
        # Column names based on NSL-KDD documentation
        col_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
        ]
        
        # Set column names
        train_data.columns = col_names
        test_data.columns = col_names
        
        # Convert attack types to binary (normal=0, attack=1)
        train_data['label'] = train_data['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
        test_data['label'] = test_data['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Combine data for consistent preprocessing
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        
        # Handle categorical features
        categorical_columns = ['protocol_type', 'service', 'flag']
        combined_data = pd.get_dummies(combined_data, columns=categorical_columns, drop_first=True)
        
        # Extract train and test data after preprocessing
        train_data = combined_data.iloc[:len(train_data)]
        test_data = combined_data.iloc[len(train_data):]
        
        # Separate features and target
        X_train = train_data.drop(['attack_type', 'difficulty', 'label'], axis=1)
        y_train = train_data['label']
        
        X_test = test_data.drop(['attack_type', 'difficulty', 'label'], axis=1)
        y_test = test_data['label']
        
        # Normalize numeric features
        scaler = StandardScaler()
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        logger.info(f"Dataset loaded and preprocessed successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return test_data, X_test.values, y_test.values, X_test.columns.tolist()
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None


def load_models():
    """Load the trained anomaly detection models."""
    try:
        models = {}
        
        # Check for Isolation Forest model
        model_path = os.path.join(MODELS_DIR, 'isolation_forest.joblib')
        if os.path.exists(model_path):
            logger.info(f"Loading Isolation Forest model from {model_path}")
            models['Isolation Forest'] = joblib.load(model_path)
        
        # Check for One-Class SVM model
        model_path = os.path.join(MODELS_DIR, 'one_class_svm.joblib')
        if os.path.exists(model_path):
            logger.info(f"Loading One-Class SVM model from {model_path}")
            models['One-Class SVM'] = joblib.load(model_path)
        
        # Check for Local Outlier Factor model
        model_path = os.path.join(MODELS_DIR, 'local_outlier_factor.joblib')
        if os.path.exists(model_path):
            logger.info(f"Loading Local Outlier Factor model from {model_path}")
            models['Local Outlier Factor'] = joblib.load(model_path)
        
        # Check for DBSCAN model
        model_path = os.path.join(MODELS_DIR, 'dbscan.joblib')
        if os.path.exists(model_path):
            logger.info(f"Loading DBSCAN model from {model_path}")
            models['DBSCAN'] = joblib.load(model_path)
        
        logger.info(f"Loaded {len(models)} models successfully")
        return models
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return {}


def evaluate_all_models(X, y, models):
    """Evaluate all models and return metrics dataframe."""
    metrics_data = []

    for name, model in models.items():
        try:
            logger.info(f"Evaluating {name}...")
            
            # Handle DBSCAN differently since it's saved as a dict
            if name == "DBSCAN":
                if isinstance(model, dict) and 'model' in model:
                    # Use DBSCAN's labels_ directly for the test data
                    # For simplicity, just apply the same labels as training
                    if 'labels' in model:
                        y_pred = np.array([1 if label == -1 else 0 for label in model['labels']])
                    else:
                        # If no labels stored, we'll have to run the model on test data
                        from sklearn.cluster import DBSCAN
                        # Create a new DBSCAN model with same parameters
                        eps = model['model'].get_params()['eps']
                        min_samples = model['model'].get_params()['min_samples']
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        dbscan.fit(X)
                        y_pred = np.array([1 if label == -1 else 0 for label in dbscan.labels_])
                else:
                    # If model is the DBSCAN object itself
                    model.fit(X)
                    y_pred = np.array([1 if label == -1 else 0 for label in model.labels_])
            
            # For Isolation Forest, One-Class SVM, LOF
            else:
                # Get predictions
                y_pred = model.predict(X)
                
                # Convert prediction format
                # For these models, -1 is anomaly (our label 1), 1 is normal (our label 0)
                if name in ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]:
                    # Convert -1/1 to 1/0 (anomaly/normal)
                    y_pred = np.array([1 if label == -1 else 0 for label in y_pred])
            
            # Calculate metrics
            metrics = calculate_model_metrics(y, y_pred)
            metrics['model'] = name
            
            # Add to metrics data
            metrics_data.append(metrics)
            
            # Generate confusion matrix visualization
            generate_confusion_matrix(y, y_pred, name)
            
            # Generate ROC curve if method supports decision function
            if hasattr(model, "decision_function"):
                # For anomaly detectors, we need to negate the scores
                # because higher decision_function values mean more normal (less anomalous)
                y_score = -model.decision_function(X)  
                generate_roc_curve(y, y_score, name)
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Convert to DataFrame
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        
        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv'), index=False)
        logger.info(f"Model metrics saved to {os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv')}")
        
        return metrics_df
    else:
        logger.warning("No metrics data collected.")
        return pd.DataFrame()


def calculate_model_metrics(y_true, y_pred):
    """Calculate evaluation metrics for a model."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    metrics = {}
    
    try:
        # Calculate basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        logger.info(f"Calculated metrics: Accuracy={metrics['accuracy']:.4f}, F1-score={metrics['f1_score']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}


def generate_confusion_matrix(y_true, y_pred, model_name):
    """Generate confusion matrix visualization."""
    try:
        from sklearn.metrics import confusion_matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        filepath = os.path.join(VISUALIZATIONS_DIR, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Confusion matrix for {model_name} saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error generating confusion matrix for {model_name}: {e}")


def generate_roc_curve(y_true, y_score, model_name):
    """Generate ROC curve visualization."""
    try:
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
        filepath = os.path.join(VISUALIZATIONS_DIR, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"ROC curve for {model_name} saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error generating ROC curve for {model_name}: {e}")


def generate_model_comparison_visualizations(metrics_df):
    """Generate visualizations comparing all models."""
    try:
        # Check if we have data
        if metrics_df.empty:
            logger.warning("No metrics data available for comparison visualizations.")
            return
        
        logger.info("Generating model comparison visualizations...")
        
        # 1. Bar chart comparing key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'specificity' in metrics_df.columns:
            key_metrics.append('specificity')
        
        plt.figure(figsize=(12, 8))
        metrics_df.set_index('model')[key_metrics].plot(kind='bar', figsize=(12, 8))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison_barchart.png'), dpi=300)
        plt.close()
        logger.info("Generated model comparison bar chart")
        
        # 2. Radar chart for multi-metric visualization
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
        
        # Function to create radar chart
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
        radar_fig = radar_chart(metrics_df, key_metrics, title='Model Comparison - Multiple Metrics')
        radar_fig.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison_radar.png'), dpi=300, bbox_inches='tight')
        plt.close(radar_fig)
        logger.info("Generated radar chart for model comparison")
        
        # 3. Line charts for individual metrics
        for metric in ['f1_score', 'precision', 'recall']:
            if metric in metrics_df.columns:
                plt.figure(figsize=(10, 6))
                metrics_df.plot(x='model', y=metric, kind='line', marker='o', figsize=(10, 6))
                plt.title(f'{metric.replace("_", " ").title()} Comparison')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.xlabel('Model')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'model_comparison_line_{metric}.png'), dpi=300)
                plt.close()
                logger.info(f"Generated line chart for {metric} comparison")
        
        # 4. Scatter plots for pairs of metrics
        # Precision vs Recall
        plt.figure(figsize=(8, 6))
        plt.scatter(metrics_df['precision'], metrics_df['recall'], s=50)
        for i, model in enumerate(metrics_df['model']):
            plt.annotate(model, (metrics_df['precision'].iloc[i], metrics_df['recall'].iloc[i]), 
                         fontsize=10, ha='right', va='bottom')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision vs Recall')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison_scatter_precision_recall.png'), dpi=300)
        plt.close()
        logger.info("Generated scatter plot for precision vs recall")
        
        # F1 Score vs Accuracy
        plt.figure(figsize=(8, 6))
        plt.scatter(metrics_df['f1_score'], metrics_df['accuracy'], s=50)
        for i, model in enumerate(metrics_df['model']):
            plt.annotate(model, (metrics_df['f1_score'].iloc[i], metrics_df['accuracy'].iloc[i]), 
                         fontsize=10, ha='right', va='bottom')
        plt.xlabel('F1 Score')
        plt.ylabel('Accuracy')
        plt.title('F1 Score vs Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison_scatter_f1_accuracy.png'), dpi=300)
        plt.close()
        logger.info("Generated scatter plot for F1 score vs accuracy")
        
        # 5. Boxplot of metrics
        # Reshape data for boxplot
        melted_df = pd.melt(metrics_df, id_vars=['model'], value_vars=key_metrics, 
                           var_name='Metric', value_name='Value')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Metric', y='Value', data=melted_df)
        plt.title('Distribution of Model Metrics')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_metrics_boxplot.png'), dpi=300)
        plt.close()
        logger.info("Generated boxplot for metric distribution")
        
        # 6. Heatmap of metrics
        plt.figure(figsize=(12, 8))
        heatmap_df = metrics_df.set_index('model')[key_metrics]
        sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('Model Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_metrics_heatmap.png'), dpi=300)
        plt.close()
        logger.info("Generated heatmap for model metrics")
        
        logger.info("All comparison visualizations generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating comparison visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """Main function for model evaluation."""
    try:
        logger.info("Starting model evaluation...")
        
        # Load dataset
        data, X, y, feature_names = load_dataset()
        if data is None or X is None or y is None:
            logger.error("Failed to load dataset. Exiting.")
            return
        
        # Load models
        models = load_models()
        if not models:
            logger.error("No models found. Exiting.")
            return
        
        # Evaluate all models
        metrics_df = evaluate_all_models(X, y, models)
        
        # Generate comparison visualizations
        if not metrics_df.empty:
            generate_model_comparison_visualizations(metrics_df)
        
        logger.info("Model evaluation completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
