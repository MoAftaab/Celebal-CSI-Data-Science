#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Model Training
-----------------------------------------
This script implements and trains various unsupervised anomaly detection models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from time import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_training')

# ML models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Deep learning
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Model, Sequential  # type: ignore
    from tensorflow.keras.layers import Dense, Input, Dropout  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available. Autoencoder model will not be available.")

# Custom imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualizations')

# Create directories if they don't exist
for dir_path in [MODELS_DIR, VISUALIZATIONS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")


def load_preprocessed_data():
    """
    Load the preprocessed data.
    
    Returns:
    --------
    dict
        Dictionary containing the preprocessed data splits
    """
    logger.info("Loading preprocessed data...")
    
    try:
        # Load X_train and X_test
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
        
        # Try to load y_train and y_test (they might not exist)
        try:
            y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
            y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
        except FileNotFoundError:
            logger.warning("No labels found. Working with unlabeled data.")
            y_train = None
            y_test = None
        
        # Load the scaler
        scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'scaler.joblib'))
        
        # Load feature names
        with open(os.path.join(PROCESSED_DATA_DIR, 'feature_names.txt'), 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': feature_names
        }
        
        logger.info(f"Data loaded successfully. X_train shape: {X_train.shape}")
        return data_dict
    
    except FileNotFoundError as e:
        logger.error(f"Error loading preprocessed data: {e}")
        logger.info("Please run data_preprocessing.py first.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        return None


class IsolationForestModel:
    """Isolation Forest model for anomaly detection."""
    
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=42):
        """
        Initialize the Isolation Forest model.
        
        Parameters:
        -----------
        n_estimators : int
            Number of base estimators in the ensemble
        max_samples : int or float
            Number of samples to draw for each base estimator
        contamination : float or 'auto'
            Expected proportion of outliers in the data set
        random_state : int
            Random state for reproducibility
        """
        self.name = "Isolation Forest"
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )
        self.params = {
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'contamination': contamination,
            'random_state': random_state
        }
    
    def fit(self, X_train):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        
        Returns:
        --------
        self
        """
        start_time = time()
        self.model.fit(X_train)
        self.training_time = time() - start_time
        logger.info(f"{self.name} trained in {self.training_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict
        
        Returns:
        --------
        numpy.ndarray
            Binary predictions (1: normal, -1: anomaly)
        """
        return self.model.predict(X)
    
    def decision_function(self, X):
        """
        Get anomaly scores.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to score
        
        Returns:
        --------
        numpy.ndarray
            Anomaly scores (lower = more anomalous)
        """
        return self.model.decision_function(X)
    
    def get_anomaly_scores(self, X):
        """
        Get normalized anomaly scores (higher = more anomalous).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to score
        
        Returns:
        --------
        numpy.ndarray
            Normalized anomaly scores (higher = more anomalous)
        """
        # For Isolation Forest, lower scores are more anomalous, so we negate them
        return -self.model.decision_function(X)
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"{self.name} saved to {filepath}")


class OneClassSVMModel:
    """One-Class SVM model for anomaly detection."""
    
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale'):
        """
        Initialize the One-Class SVM model.
        
        Parameters:
        -----------
        kernel : str
            Kernel type
        nu : float
            Upper bound on the fraction of training errors / lower bound on the fraction of support vectors
        gamma : float or 'scale' or 'auto'
            Kernel coefficient
        """
        self.name = "One-Class SVM"
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.params = {
            'kernel': kernel,
            'nu': nu,
            'gamma': gamma
        }
    
    def fit(self, X_train):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        
        Returns:
        --------
        self
        """
        start_time = time()
        self.model.fit(X_train)
        self.training_time = time() - start_time
        logger.info(f"{self.name} trained in {self.training_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict
        
        Returns:
        --------
        numpy.ndarray
            Binary predictions (1: normal, -1: anomaly)
        """
        return self.model.predict(X)
    
    def decision_function(self, X):
        """
        Get anomaly scores.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to score
        
        Returns:
        --------
        numpy.ndarray
            Anomaly scores (lower = more anomalous)
        """
        return self.model.decision_function(X)
    
    def get_anomaly_scores(self, X):
        """
        Get normalized anomaly scores (higher = more anomalous).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to score
        
        Returns:
        --------
        numpy.ndarray
            Normalized anomaly scores (higher = more anomalous)
        """
        # For One-Class SVM, lower scores are more anomalous, so we negate them
        return -self.model.decision_function(X)
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"{self.name} saved to {filepath}")


class LocalOutlierFactorModel:
    """Local Outlier Factor model for anomaly detection."""
    
    def __init__(self, n_neighbors=20, contamination='auto', novelty=True):
        """
        Initialize the Local Outlier Factor model.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to use
        contamination : float or 'auto'
            Expected proportion of outliers in the data set
        novelty : bool
            If True, LOF is used for novelty detection
        """
        self.name = "Local Outlier Factor"
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty
        )
        self.params = {
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'novelty': novelty
        }
    
    def fit(self, X_train):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        
        Returns:
        --------
        self
        """
        start_time = time()
        self.model.fit(X_train)
        self.training_time = time() - start_time
        logger.info(f"{self.name} trained in {self.training_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict
        
        Returns:
        --------
        numpy.ndarray
            Binary predictions (1: normal, -1: anomaly)
        """
        return self.model.predict(X)
    
    def decision_function(self, X):
        """
        Get anomaly scores.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to score
        
        Returns:
        --------
        numpy.ndarray
            Anomaly scores (lower = more anomalous)
        """
        return self.model.decision_function(X)
    
    def get_anomaly_scores(self, X):
        """
        Get normalized anomaly scores (higher = more anomalous).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to score
        
        Returns:
        --------
        numpy.ndarray
            Normalized anomaly scores (higher = more anomalous)
        """
        # For LOF, lower scores are more anomalous, so we negate them
        return -self.model.decision_function(X)
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"{self.name} saved to {filepath}")


class DBSCANModel:
    """DBSCAN model for anomaly detection."""
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize the DBSCAN model.
        
        Parameters:
        -----------
        eps : float
            Maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples : int
            Number of samples in a neighborhood for a point to be considered as a core point
        """
        self.name = "DBSCAN"
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.params = {
            'eps': eps,
            'min_samples': min_samples
        }
    
    def fit(self, X_train):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        
        Returns:
        --------
        self
        """
        start_time = time()
        self.model.fit(X_train)
        self.training_time = time() - start_time
        logger.info(f"{self.name} trained in {self.training_time:.2f} seconds")
        
        # Store training data and labels
        self.X_train = X_train
        self.labels_ = self.model.labels_
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict
        
        Returns:
        --------
        numpy.ndarray
            Binary predictions (1: normal, -1: anomaly)
        """
        # DBSCAN doesn't have a predict method, so we need to implement it
        # Points with label -1 are outliers/anomalies
        # Convert labels: -1 (outlier) -> -1, others -> 1 (inlier)
        return np.where(self.model.fit_predict(X) == -1, -1, 1)
    
    def get_anomaly_scores(self, X):
        """
        Get normalized anomaly scores (higher = more anomalous).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to score
        
        Returns:
        --------
        numpy.ndarray
            Normalized anomaly scores (higher = more anomalous)
        """
        # DBSCAN doesn't naturally provide anomaly scores, so we compute them
        # based on distance to nearest core point
        
        # For each point in X, compute the minimum distance to any core point in the training set
        core_samples_mask = np.zeros_like(self.labels_, dtype=bool)
        core_samples_mask[self.model.core_sample_indices_] = True
        core_samples = self.X_train[core_samples_mask]
        
        # If there are no core samples, all points are considered anomalies
        if len(core_samples) == 0:
            return np.ones(X.shape[0])
        
        # Compute the minimum distance to any core point for each point in X
        distances = np.zeros(X.shape[0])
        for i, point in enumerate(X):
            # Compute distances to all core points
            point_distances = np.sqrt(np.sum((core_samples - point) ** 2, axis=1))
            # Get minimum distance
            distances[i] = np.min(point_distances)
        
        return distances
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"{self.name} saved to {filepath}")


class AutoencoderModel:
    """Autoencoder model for anomaly detection."""
    
    def __init__(self, input_dim, encoding_dim=10, hidden_layers=[32, 16], dropout_rate=0.2, 
                 activation='relu', learning_rate=0.001, epochs=100, batch_size=32, validation_split=0.1):
        """Initialize the Autoencoder model."""
        self.name = "Autoencoder"
        
        # Check if TensorFlow is available
        if not HAS_TENSORFLOW:
            logger.error("TensorFlow not available. Autoencoder model cannot be used.")
            raise ImportError("TensorFlow not available. Please install TensorFlow to use the Autoencoder model.")
        """
        Initialize the Autoencoder model.
        
        Parameters:
        -----------
        input_dim : int
            Input dimension (number of features)
        encoding_dim : int
            Dimension of the encoding layer
        hidden_layers : list
            List of neurons in each hidden layer
        dropout_rate : float
            Dropout rate for regularization
        activation : str
            Activation function
        learning_rate : float
            Learning rate for the optimizer
        epochs : int
            Number of epochs for training
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of the training data to be used as validation data
        """
        self.name = "Autoencoder"
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = self._build_model()
        self.params = {
            'input_dim': input_dim,
            'encoding_dim': encoding_dim,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'activation': activation,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split
        }
    
    def _build_model(self):
        """
        Build the autoencoder model.
        
        Returns:
        --------
        tensorflow.keras.Model
            The autoencoder model
        """
        # Input layer
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoder = input_layer
        for layer_size in self.hidden_layers:
            encoder = Dense(layer_size, activation=self.activation)(encoder)
            encoder = Dropout(self.dropout_rate)(encoder)
        
        # Bottleneck layer
        bottleneck = Dense(self.encoding_dim, activation=self.activation, name='bottleneck')(encoder)
        
        # Decoder
        decoder = bottleneck
        for layer_size in reversed(self.hidden_layers):
            decoder = Dense(layer_size, activation=self.activation)(decoder)
            decoder = Dropout(self.dropout_rate)(decoder)
        
        # Output layer
        output_layer = Dense(self.input_dim, activation='linear')(decoder)
        
        # Create the autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile the model
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return autoencoder
    
    def fit(self, X_train):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        
        Returns:
        --------
        self
        """
        # Create callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Create a model checkpoint callback
        checkpoint_path = os.path.join(MODELS_DIR, 'autoencoder_checkpoint.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train the model
        start_time = time()
        history = self.model.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        self.training_time = time() - start_time
        
        self.history = history.history
        
        # Compute the reconstruction error threshold based on training data
        self._set_threshold(X_train)
        
        logger.info(f"{self.name} trained in {self.training_time:.2f} seconds")
        return self
    
    def _set_threshold(self, X_train, percentile=95):
        """
        Set the anomaly threshold based on the reconstruction error of the training data.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        percentile : int or float
            Percentile to use for threshold
        """
        # Get reconstruction errors on training data
        reconstructions = self.model.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        
        # Set threshold at the specified percentile
        self.threshold = np.percentile(mse, percentile)
        logger.info(f"Anomaly threshold set at {self.threshold:.6f} (percentile: {percentile})")
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict
        
        Returns:
        --------
        numpy.ndarray
            Binary predictions (1: normal, -1: anomaly)
        """
        anomaly_scores = self.get_anomaly_scores(X)
        return np.where(anomaly_scores > self.threshold, -1, 1)
    
    def get_anomaly_scores(self, X):
        """
        Get anomaly scores (reconstruction errors).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to score
        
        Returns:
        --------
        numpy.ndarray
            Anomaly scores (higher = more anomalous)
        """
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        self.model.save(filepath)
        
        # Save additional information
        model_info = {
            'threshold': self.threshold,
            'params': self.params,
            'history': self.history
        }
        joblib.dump(model_info, f"{filepath}_info.joblib")
        
        logger.info(f"{self.name} saved to {filepath}")


def evaluate_model(model, X_test, y_test=None):
    """
    Evaluate a model on test data.
    
    Parameters:
    -----------
    model : object
        Trained model with predict and get_anomaly_scores methods
    X_test : numpy.ndarray
        Test data
    y_test : numpy.ndarray or None
        Test labels (0: normal, 1: anomaly) or None for unlabeled data
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating {model.name}...")
    
    # Start timer for prediction time measurement
    start_time = time()
    
    # Get predictions and anomaly scores
    if hasattr(model, 'predict'):
        # For models with a predict method
        y_pred_raw = model.predict(X_test)
        # Convert from (-1, 1) to (1, 0) for anomaly labels
        y_pred = np.where(y_pred_raw == -1, 1, 0)
    else:
        # Fallback for models without a predict method
        anomaly_scores = model.get_anomaly_scores(X_test)
        # Use median as threshold if not explicitly set
        threshold = getattr(model, 'threshold', np.median(anomaly_scores))
        y_pred = np.where(anomaly_scores > threshold, 1, 0)
    
    # Get anomaly scores (higher = more anomalous)
    anomaly_scores = model.get_anomaly_scores(X_test)
    
    # Measure prediction time
    prediction_time = time() - start_time
    
    # Initialize results dictionary
    results = {
        'model_name': model.name,
        'prediction_time': prediction_time,
        'training_time': getattr(model, 'training_time', None),
        'params': getattr(model, 'params', {})
    }
    
    # If we have ground truth labels
    if y_test is not None:
        # Convert ground truth to binary (0: normal, 1: anomaly)
        y_true = y_test.astype(int)
        
        # Compute classification metrics
        results.update({
            'accuracy': np.mean(y_pred == y_true),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        })
        
        # Compute ROC AUC if we have continuous anomaly scores
        if anomaly_scores is not None:
            results['roc_auc'] = roc_auc_score(y_true, anomaly_scores)
            results['pr_auc'] = average_precision_score(y_true, anomaly_scores)
    
    logger.info(f"Evaluation of {model.name} completed.")
    
    # Print evaluation results
    if y_test is not None:
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1_score']:.4f}")
        if 'roc_auc' in results:
            logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
            logger.info(f"PR AUC: {results['pr_auc']:.4f}")
    
    logger.info(f"Prediction time: {prediction_time:.4f} seconds")
    
    return results


def visualize_results(results, y_test=None, X_test=None):
    """
    Visualize model evaluation results.
    
    Parameters:
    -----------
    results : list
        List of model evaluation result dictionaries
    y_test : numpy.ndarray or None
        Test labels (0: normal, 1: anomaly) or None for unlabeled data
    X_test : numpy.ndarray or None
        Test data (used for dimension reduction visualization)
    
    Returns:
    --------
    None
    """
    logger.info("Visualizing model evaluation results...")
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Extract model names and metrics
    model_names = [result['model_name'] for result in results]
    
    # Metrics visualization (for labeled data)
    if y_test is not None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Add ROC AUC if available
        if 'roc_auc' in results[0]:
            metrics.append('roc_auc')
            metrics.append('pr_auc')
        
        # Create metrics comparison bar chart
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(model_names))
        width = 0.15
        multiplier = 0
        
        for metric in metrics:
            offset = width * multiplier
            metric_values = [result[metric] for result in results]
            
            plt.bar(x + offset, metric_values, width, label=metric)
            
            multiplier += 1
        
        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * (len(metrics) - 1) / 2, model_names, rotation=45, ha='right')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        # Save the plot
        metrics_plot_path = os.path.join(VISUALIZATIONS_DIR, 'metrics_comparison.png')
        plt.savefig(metrics_plot_path)
        plt.close()
        logger.info(f"Metrics comparison plot saved to {metrics_plot_path}")
        
        # Create a heatmap of metrics for each model
        metrics_data = []
        for result in results:
            metrics_data.append([result[metric] for metric in metrics])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics_data, annot=True, cmap='YlGnBu', xticklabels=metrics, yticklabels=model_names)
        plt.title('Model Performance Metrics Heatmap')
        plt.tight_layout()
        
        # Save the plot
        metrics_heatmap_path = os.path.join(VISUALIZATIONS_DIR, 'metrics_heatmap.png')
        plt.savefig(metrics_heatmap_path)
        plt.close()
        logger.info(f"Metrics heatmap saved to {metrics_heatmap_path}")
    
    # Training and prediction time comparison
    plt.figure(figsize=(12, 6))
    
    # Extract times
    training_times = [result.get('training_time', 0) or 0 for result in results]
    prediction_times = [result.get('prediction_time', 0) or 0 for result in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, training_times, width, label='Training Time (s)')
    plt.bar(x + width/2, prediction_times, width, label='Prediction Time (s)')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.title('Model Training and Prediction Time Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    time_plot_path = os.path.join(VISUALIZATIONS_DIR, 'time_comparison.png')
    plt.savefig(time_plot_path)
    plt.close()
    logger.info(f"Time comparison plot saved to {time_plot_path}")
    
    # Save results to CSV for easy reference
    results_df = pd.DataFrame(results)
    
    # Drop complex columns that can't be easily saved to CSV
    for col in ['confusion_matrix', 'params']:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])
    
    # Save to CSV
    results_csv_path = os.path.join(VISUALIZATIONS_DIR, 'model_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to CSV: {results_csv_path}")


def main():
    """Main function to execute the model training pipeline."""
    logger.info("Starting model training pipeline...")
    
    # Load preprocessed data
    data = load_preprocessed_data()
    
    if data is None:
        logger.error("Failed to load preprocessed data. Exiting.")
        return
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Define models to train
    input_dim = X_train.shape[1]
    
    models = [
        IsolationForestModel(n_estimators=100, contamination='auto'),
        OneClassSVMModel(kernel='rbf', nu=0.05),
        LocalOutlierFactorModel(n_neighbors=20, contamination='auto'),
        DBSCANModel(eps=0.5, min_samples=5),
        AutoencoderModel(input_dim=input_dim, encoding_dim=8, hidden_layers=[32, 16], 
                         epochs=50, batch_size=64)
    ]
    
    # Train and evaluate models
    results = []
    
    for model in models:
        # Train model
        logger.info(f"Training {model.name}...")
        model.fit(X_train)
        
        # Evaluate model
        model_results = evaluate_model(model, X_test, y_test)
        results.append(model_results)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{model.name.lower().replace(' ', '_')}.joblib")
        model.save(model_path)
    
    # Visualize results
    visualize_results(results, y_test, X_test)
    
    # Find the best model based on F1 score (if labeled data is available)
    if y_test is not None:
        best_model_idx = np.argmax([result['f1_score'] for result in results])
        best_model_name = results[best_model_idx]['model_name']
        
        logger.info(f"Best model: {best_model_name} with F1 score: {results[best_model_idx]['f1_score']:.4f}")
        
        # Create a symlink to the best model
        best_model_path = os.path.join(MODELS_DIR, f"{best_model_name.lower().replace(' ', '_')}.joblib")
        best_model_link = os.path.join(MODELS_DIR, "best_model.joblib")
        
        # On Windows, we need to copy the file instead of creating a symlink
        import platform
        if platform.system() == 'Windows':
            import shutil
            shutil.copy2(best_model_path, best_model_link)
        else:
            # On Unix-like systems, we can create a symlink
            if os.path.exists(best_model_link):
                os.remove(best_model_link)
            os.symlink(best_model_path, best_model_link)
        
        logger.info(f"Best model saved as {best_model_link}")
    
    logger.info("Model training pipeline completed successfully.")


if __name__ == "__main__":
    main() 
