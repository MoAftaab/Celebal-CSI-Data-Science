#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Model Generation
------------------------------------------
This script generates and saves anomaly detection models.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_generation')

# Define paths
DATA_DIR = "data"
MODELS_DIR = "models"
VISUALIZATIONS_DIR = "visualizations"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def load_nsl_kdd_data():
    """Load and preprocess the NSL-KDD dataset."""
    try:
        # Use full KDDTrain+ dataset instead of the small sample
        file_path = os.path.join(DATA_DIR, "datasets", "KDDTrain+.csv")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}. Please run download_dataset.py first.")
            return None
        
        logger.info(f"Loading NSL-KDD dataset from {file_path}")
        
        # Load the data
        # NSL-KDD has no headers by default
        data = pd.read_csv(file_path, header=None)
        
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
        data.columns = col_names
        
        # Convert attack types to binary (normal=0, attack=1)
        data['label'] = data['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading NSL-KDD data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def preprocess_nsl_kdd_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the NSL-KDD dataset for anomaly detection.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The NSL-KDD dataframe to preprocess
    test_size : float
        Proportion of the dataset to be used as test set
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing preprocessed data splits
    """
    try:
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Handle categorical features
        categorical_columns = ['protocol_type', 'service', 'flag']
        processed_df = pd.get_dummies(processed_df, columns=categorical_columns, drop_first=True)
        
        # Separate features and target
        y = processed_df['label'].values
        X_df = processed_df.drop(['attack_type', 'difficulty', 'label'], axis=1)
        
        # Handle missing values - replace NaN with column mean
        X_df = X_df.fillna(X_df.mean())
        
        # If there are still NaN values (e.g., in columns that are all NaN), replace with 0
        X_df = X_df.fillna(0)
        
        # Convert to numpy array
        X = X_df.values
        
        # Keep track of original indices
        indices = np.arange(len(X))
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X, y, indices, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the feature names
        feature_names = X_df.columns.tolist()
        feature_names_path = os.path.join(DATA_DIR, 'processed', 'feature_names.txt')
        os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(feature_names))
        logger.info(f"Feature names saved to {feature_names_path}")
        
        # Save the scaler
        scaler_path = os.path.join(DATA_DIR, 'processed', 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save the processed data
        np.save(os.path.join(DATA_DIR, 'processed', 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(DATA_DIR, 'processed', 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(DATA_DIR, 'processed', 'y_train.npy'), y_train)
        np.save(os.path.join(DATA_DIR, 'processed', 'y_test.npy'), y_test)
        logger.info(f"Processed data saved to {os.path.join(DATA_DIR, 'processed')}")
        
        # Create a dictionary with processed data
        data_dict = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'scaler': scaler,
            'feature_names': feature_names
        }
        
        logger.info(f"NSL-KDD data preprocessed successfully. X_train shape: {X_train_scaled.shape}")
        return data_dict
        
    except Exception as e:
        logger.error(f"Error preprocessing NSL-KDD data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def train_isolation_forest(X_train, contamination=0.1, random_state=42):
    """Train an Isolation Forest model for anomaly detection."""
    try:
        # Create and train the model with optimized parameters
        model = IsolationForest(
            n_estimators=300,  # Increased for better robustness
            max_samples=min(256, X_train.shape[0]),  # Explicit sample size for better modeling
            contamination=contamination,  # Proportion of outliers in the data
            max_features=0.8,  # Use a subset of features for better generalization
            bootstrap=True,  # Use bootstrap samples
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        # Fit the model
        model.fit(X_train)
        
        return model
    except Exception as e:
        logger.error(f"Error training Isolation Forest: {e}")
        return None


def train_one_class_svm(X_train, contamination=0.1):
    """Train a One-Class SVM model for anomaly detection."""
    try:
        # Estimate nu parameter from contamination
        nu = contamination
        
        # Create and train the model with optimized parameters
        model = OneClassSVM(
            nu=nu,  # Upper bound on the fraction of training errors
            kernel='rbf',
            gamma='auto',  # Changed to auto which often works better with large datasets
            shrinking=True,  # Use shrinking heuristic for faster training
            tol=1e-3,  # Tolerance for stopping criterion
            cache_size=2000  # Increased cache size for faster computation
        )
        
        # Fit the model
        model.fit(X_train)
        
        return model
    except Exception as e:
        logger.error(f"Error training One-Class SVM: {e}")
        return None


def generate_isolation_forest(X_train):
    """Generate and save an Isolation Forest model."""
    try:
        logger.info("Training Isolation Forest model...")
        # Adjust contamination based on the actual distribution in the dataset
        # Using a higher value since we know the NSL-KDD dataset has many attack samples
        model = train_isolation_forest(X_train, contamination=0.35)
        
        # Save the model
        model_path = os.path.join(MODELS_DIR, 'isolation_forest.joblib')
        joblib.dump(model, model_path)
        logger.info(f"Isolation Forest model saved to {model_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error generating Isolation Forest model: {e}")
        return None


def generate_one_class_svm(X_train):
    """Generate and save a One-Class SVM model."""
    try:
        logger.info("Training One-Class SVM model...")
        # Adjust contamination based on the actual distribution in the dataset
        model = train_one_class_svm(X_train, contamination=0.35)
        
        # Save the model
        model_path = os.path.join(MODELS_DIR, 'one_class_svm.joblib')
        joblib.dump(model, model_path)
        logger.info(f"One-Class SVM model saved to {model_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error generating One-Class SVM model: {e}")
        return None


def generate_local_outlier_factor(X_train):
    """Generate and save a Local Outlier Factor model."""
    try:
        logger.info("Training Local Outlier Factor model...")
        # Increase n_neighbors for better stability and contamination for actual distribution
        model = LocalOutlierFactor(
            n_neighbors=30,  # Increased from 20 to 30
            contamination=0.35,
            novelty=True,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train)
        
        # Save the model
        model_path = os.path.join(MODELS_DIR, 'local_outlier_factor.joblib')
        joblib.dump(model, model_path)
        logger.info(f"Local Outlier Factor model saved to {model_path}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error generating Local Outlier Factor model: {e}")
        return None


def generate_dbscan(X_train, y_train):
    """Generate and save a DBSCAN model."""
    try:
        logger.info("Training DBSCAN model...")
        
        # Create a DBSCAN model with optimized parameters
        # Use a subset of data for parameter estimation if the dataset is too large
        X_sample = X_train
        if len(X_train) > 10000:
            from sklearn.utils import resample
            # Use stratified sampling to ensure we get both normal and anomaly samples
            normal_indices = np.where(y_train == 0)[0]
            anomaly_indices = np.where(y_train == 1)[0]
            
            # Sample from both classes
            n_normal = min(5000, len(normal_indices))
            n_anomaly = min(5000, len(anomaly_indices))
            
            normal_sample_indices = np.random.choice(normal_indices, size=n_normal, replace=False)
            anomaly_sample_indices = np.random.choice(anomaly_indices, size=n_anomaly, replace=False)
            
            # Combine samples
            sample_indices = np.concatenate([normal_sample_indices, anomaly_sample_indices])
            X_sample = X_train[sample_indices]
            y_sample = y_train[sample_indices]
        
        # Calculate average distance to 5th nearest neighbor to estimate eps
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=10)  # Increased from 5 to 10 for better stability
        nn.fit(X_sample)
        distances, _ = nn.kneighbors(X_sample)
        
        # Calculate different percentiles of distances to find a good eps value
        # This helps to handle different density regions
        p50 = np.percentile(distances[:, -1], 50)
        p75 = np.percentile(distances[:, -1], 75)
        
        # Choose a smaller eps value to detect smaller clusters (anomalies)
        eps = p50 * 0.5
        logger.info(f"DBSCAN eps parameter calculated: {eps}")
        
        # Train DBSCAN with optimized parameters
        model = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)  # Reduced min_samples for better sensitivity
        cluster_labels = model.fit_predict(X_train)
        
        # Map DBSCAN clusters to normal/anomaly labels
        # First, count the distribution of real labels in each cluster
        clusters = np.unique(cluster_labels)
        cluster_to_label = {}
        
        # For each cluster, assign the majority label (normal=0 or anomaly=1)
        for cluster in clusters:
            if cluster == -1:  # Noise points are considered anomalies
                cluster_to_label[cluster] = 1
            else:
                # Count normal vs anomaly in this cluster
                mask = (cluster_labels == cluster)
                normal_count = np.sum(y_train[mask] == 0)
                anomaly_count = np.sum(y_train[mask] == 1)
                # Assign majority label
                cluster_to_label[cluster] = 0 if normal_count > anomaly_count else 1
        
        # Create a function to map clusters to binary labels
        def cluster_to_binary(cluster):
            return cluster_to_label.get(cluster, 1)  # Default to anomaly if cluster not seen
        
        # Store model, data and mapping function
        model_with_data = {
            'model': model,
            'cluster_to_label': cluster_to_label,
            'eps': eps,
            'min_samples': 5,  # Updated to match the value used in model creation
            'X_train': X_train[:5000] if len(X_train) > 5000 else X_train,  # Keep only a subset to save memory
        }
        
        # Save the model
        model_path = os.path.join(MODELS_DIR, 'dbscan.joblib')
        joblib.dump(model_with_data, model_path)
        logger.info(f"DBSCAN model saved to {model_path}")
        
        return model_with_data
    
    except Exception as e:
        logger.error(f"Error generating DBSCAN model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def calculate_model_metrics(y_true, y_pred):
    """
    Calculate various metrics for model evaluation.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {}
    
    try:
        # Calculate metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate ROC AUC if possible
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['roc_auc'] = None
            
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics


def generate_model_results_csv(models, X_test, y_test, threshold=0.5):
    """Generate and save a CSV file with model results."""
    try:
        logger.info("Generating model results CSV...")
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            if name == "DBSCAN":
                # For DBSCAN, predict with the trained model
                try:
                    # Get the model's cluster predictions for test data
                    cluster_predictions = model['model'].fit_predict(X_test)
                    
                    # Map clusters to binary labels using our mapping function
                    y_pred = np.zeros_like(cluster_predictions)
                    for i, cluster in enumerate(cluster_predictions):
                        y_pred[i] = model['cluster_to_label'].get(cluster, 1)  # Default to anomaly
                    
                except Exception as e:
                    logger.error(f"Error calculating DBSCAN predictions: {e}")
                    continue
                
            else:
                # For other models, predict on test data
                y_pred = model.predict(X_test)
                # Convert to binary (normal=0, anomaly=1)
                if name in ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]:
                    y_pred = np.array([1 if label == -1 else 0 for label in y_pred])
            
            # Calculate metrics
            metrics = calculate_model_metrics(y_test, y_pred)
            metrics['model_name'] = name
            
            # Log results
            logger.info(f"{name} - Accuracy: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1_score', 0):.4f}")
            
            results.append(metrics)
        
        # Create DataFrame and save to CSV
        if results:
            results_df = pd.DataFrame(results)
            output_path = os.path.join(VISUALIZATIONS_DIR, "model_results.csv")
            results_df.to_csv(output_path, index=False)
            logger.info(f"Model results saved to {output_path}")
            
            # Create a simple comparison plot
            plt.figure(figsize=(12, 8))
            
            # Set up metrics to plot
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            if 'roc_auc' in results_df.columns:
                metrics_with_values = results_df.dropna(subset=['roc_auc'])
                if not metrics_with_values.empty:
                    metrics_to_plot.append('roc_auc')
            
            # Create the plot
            ax = results_df.plot(x='model_name', y=metrics_to_plot, kind='bar', figsize=(12, 8))
            plt.title("Model Performance Comparison")
            plt.xlabel("Model")
            plt.ylabel("Score")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(VISUALIZATIONS_DIR, "model_comparison.png")
            plt.savefig(plot_path, dpi=300)
            logger.info(f"Model comparison plot saved to {plot_path}")
            
            return results_df
        else:
            logger.error("No results to save.")
            return None
            
    except Exception as e:
        logger.error(f"Error generating model results CSV: {e}")
        return None


def main():
    """Main function to generate models."""
    logger.info("Starting model generation...")
    
    # Load dataset
    data = load_nsl_kdd_data()
    if data is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # Preprocess data
    processed_data = preprocess_nsl_kdd_data(data)
    if processed_data is None:
        logger.error("Failed to preprocess data. Exiting.")
        return
    
    # Extract X_train and y_train for model training
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    
    # Create directory for models
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Generate and store models
    logger.info("Generating models...")
    models = {}
    
    # Isolation Forest
    isolation_forest = generate_isolation_forest(X_train)
    if isolation_forest is not None:
        models['Isolation Forest'] = isolation_forest
    
    # One-Class SVM
    one_class_svm = generate_one_class_svm(X_train)
    if one_class_svm is not None:
        models['One-Class SVM'] = one_class_svm
    
    # Local Outlier Factor
    local_outlier_factor = generate_local_outlier_factor(X_train)
    if local_outlier_factor is not None:
        models['Local Outlier Factor'] = local_outlier_factor
    
    # DBSCAN - now passing y_train as well
    dbscan = generate_dbscan(X_train, y_train)
    if dbscan is not None:
        models['DBSCAN'] = dbscan
    
    # Generate model results CSV
    generate_model_results_csv(
        models, 
        processed_data['X_test'], 
        processed_data['y_test'], 
        threshold=0.5
    )
    
    logger.info("Model generation completed successfully.")


if __name__ == "__main__":
    main() 