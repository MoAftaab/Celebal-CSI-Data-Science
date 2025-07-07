#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Data Preprocessing
---------------------------------------------
This script handles data loading, cleaning, and preprocessing for the network anomaly detection project.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
from io import BytesIO
import tarfile
import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_preprocessing')

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
VISUALIZATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')

# Create directories if they don't exist
for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, VISUALIZATIONS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")


def download_dataset():
    """
    Download the network telemetry dataset.
    We'll use the Cisco telemetry dataset for BGP anomaly detection.
    """
    logger.info("Downloading dataset...")
    
    # URL for the dataset (we'll use a sample one from Cisco's telemetry project)
    # For the purpose of this project, we're using dataset 2 from cisco-ie/telemetry which contains BGP anomalies
    dataset_url = "https://github.com/cisco-ie/telemetry/raw/master/2/dataset_2.csv"
    
    try:
        response = requests.get(dataset_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the dataset to the raw data directory
        dataset_path = os.path.join(RAW_DATA_DIR, 'network_telemetry.csv')
        with open(dataset_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Dataset downloaded successfully to {dataset_path}")
        return dataset_path
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading dataset: {e}")
        
        # Create a synthetic dataset for demonstration if download fails
        logger.info("Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset()


def create_synthetic_dataset(n_samples=10000, anomaly_ratio=0.05):
    """
    Create a synthetic network telemetry dataset for demonstration purposes.
    This is used as a fallback if the actual dataset cannot be downloaded.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    anomaly_ratio : float
        Ratio of anomalies in the dataset
    
    Returns:
    --------
    str
        Path to the created dataset
    """
    logger.info(f"Creating synthetic dataset with {n_samples} samples and {anomaly_ratio:.1%} anomalies")
    
    # Number of anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    # Generate normal data (multivariate Gaussian)
    np.random.seed(42)
    
    # Features that simulate network telemetry data
    features = [
        'packets_in', 'packets_out', 'bytes_in', 'bytes_out', 
        'packet_loss_rate', 'latency_ms', 'jitter_ms', 'error_rate',
        'bandwidth_utilization', 'connection_count', 'retransmission_rate',
        'cpu_utilization', 'memory_usage', 'tcp_connections', 'udp_connections'
    ]
    
    # Generate normal instances with correlated features
    mean = np.array([1000, 800, 150000, 120000, 0.02, 20, 5, 0.01, 0.65, 
                     150, 0.03, 0.45, 0.55, 100, 50])
    
    # Create a correlation matrix for realistic data
    corr = np.random.rand(len(features), len(features))
    corr = (corr + corr.T) / 2  # Make it symmetric
    np.fill_diagonal(corr, 1)  # Set diagonal to 1
    
    # Generate covariance matrix from correlation
    std = np.array([200, 150, 30000, 25000, 0.01, 5, 2, 0.005, 0.1, 
                   30, 0.01, 0.1, 0.1, 20, 10])
    cov = np.outer(std, std) * corr
    
    # Generate normal samples
    normal_data = np.random.multivariate_normal(mean, cov, n_normal)
    
    # Generate anomalies with different distributions
    anomaly_types = np.random.choice(3, n_anomalies)
    anomaly_data = np.zeros((n_anomalies, len(features)))
    
    # Type 1: High traffic anomalies (DoS-like)
    type1_indices = np.where(anomaly_types == 0)[0]
    anomaly_data[type1_indices] = np.random.multivariate_normal(
        mean * np.array([5, 0.2, 8, 0.1, 5, 3, 4, 5, 0.95, 2, 8, 0.9, 0.9, 1.5, 0.5]), 
        cov * 0.5, 
        len(type1_indices)
    )
    
    # Type 2: Low traffic anomalies (system failure-like)
    type2_indices = np.where(anomaly_types == 1)[0]
    anomaly_data[type2_indices] = np.random.multivariate_normal(
        mean * np.array([0.1, 0.1, 0.1, 0.1, 10, 10, 10, 8, 0.1, 0.1, 10, 0.1, 0.1, 0.1, 0.1]), 
        cov * 0.5, 
        len(type2_indices)
    )
    
    # Type 3: Unusual pattern anomalies (scan-like)
    type3_indices = np.where(anomaly_types == 2)[0]
    anomaly_data[type3_indices] = np.random.multivariate_normal(
        mean * np.array([1.5, 3, 0.8, 2.5, 0.5, 0.5, 2, 3, 0.5, 5, 0.5, 0.7, 0.6, 3, 0.2]), 
        cov * 0.5, 
        len(type3_indices)
    )
    
    # Combine normal and anomaly data
    X = np.vstack([normal_data, anomaly_data])
    
    # Create labels (0 for normal, 1 for anomaly)
    y = np.zeros(n_samples)
    y[n_normal:] = 1
    
    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Create a DataFrame
    df = pd.DataFrame(X, columns=features)
    df['is_anomaly'] = y
    df['anomaly_type'] = -1  # Default for normal traffic
    
    # Set anomaly types
    anomaly_indices = np.where(y == 1)[0]
    shuffled_anomaly_types = anomaly_types[np.argsort(indices[n_normal:])]
    df.loc[anomaly_indices, 'anomaly_type'] = shuffled_anomaly_types
    
    # Add timestamp
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    timestamps = timestamps[indices]  # Shuffle timestamps to match data
    df['timestamp'] = timestamps
    
    # Save to CSV
    output_path = os.path.join(RAW_DATA_DIR, 'synthetic_network_telemetry.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Synthetic dataset created and saved to {output_path}")
    return output_path


def load_and_explore_data(dataset_path):
    """
    Load and explore the dataset, generating basic visualizations.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file
    
    Returns:
    --------
    pd.DataFrame
        The loaded dataset
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        # Check file extension
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.parquet'):
            df = pd.read_parquet(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            logger.error(f"Unsupported file format: {dataset_path}")
            return None
        
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Basic data exploration
        logger.info("\nDataset Information:")
        df_info = df.info(verbose=False)
        logger.info(f"\nMissing values:\n{df.isnull().sum()}")
        logger.info(f"\nSummary statistics:\n{df.describe()}")
        
        # Check if 'is_anomaly' column exists
        if 'is_anomaly' in df.columns:
            # Count of anomalies
            anomaly_count = df['is_anomaly'].sum()
            logger.info(f"\nAnomaly count: {anomaly_count} ({anomaly_count/len(df):.2%} of the dataset)")
            
            # Create a bar plot for the distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(x='is_anomaly', data=df)
            plt.title('Distribution of Normal vs Anomaly Samples')
            plt.xlabel('Is Anomaly')
            plt.ylabel('Count')
            plt.xticks([0, 1], ['Normal', 'Anomaly'])
            
            # Save the plot
            dist_plot_path = os.path.join(VISUALIZATIONS_DIR, 'class_distribution.png')
            plt.savefig(dist_plot_path)
            plt.close()
            logger.info(f"Class distribution plot saved to {dist_plot_path}")
            
            # If there are different anomaly types
            if 'anomaly_type' in df.columns:
                # Distribution of anomaly types
                plt.figure(figsize=(12, 6))
                anomaly_df = df[df['is_anomaly'] == 1]
                sns.countplot(x='anomaly_type', data=anomaly_df)
                plt.title('Distribution of Anomaly Types')
                plt.xlabel('Anomaly Type')
                plt.ylabel('Count')
                
                # Save the plot
                anomaly_type_plot_path = os.path.join(VISUALIZATIONS_DIR, 'anomaly_type_distribution.png')
                plt.savefig(anomaly_type_plot_path)
                plt.close()
                logger.info(f"Anomaly type distribution plot saved to {anomaly_type_plot_path}")
        
        # Create correlation heatmap for numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variables from correlation analysis if they exist
        for col in ['is_anomaly', 'anomaly_type']:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        if len(numeric_cols) > 1:  # Need at least 2 columns for correlation
            plt.figure(figsize=(14, 12))
            correlation = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .5})
            plt.title('Feature Correlation Heatmap')
            
            # Save the plot
            corr_plot_path = os.path.join(VISUALIZATIONS_DIR, 'correlation_heatmap.png')
            plt.savefig(corr_plot_path)
            plt.close()
            logger.info(f"Correlation heatmap saved to {corr_plot_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading or exploring data: {e}")
        return None


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the dataset for anomaly detection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to preprocess
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing the preprocessed data splits and scaler
    """
    logger.info("Preprocessing the dataset...")
    
    try:
        # Make a copy of the dataframe
        df = df.copy()
        
        # Drop non-feature columns
        drop_cols = []
        
        # Check and handle timestamp column
        timestamp_cols = [col for col in df.columns if 'time' in col.lower()]
        if timestamp_cols:
            # Convert to datetime if not already
            for col in timestamp_cols:
                if df[col].dtype != 'datetime64[ns]':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        logger.warning(f"Could not convert {col} to datetime. Dropping it.")
                        drop_cols.append(col)
                
                # Extract useful datetime features if column is valid
                if col not in drop_cols:
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    drop_cols.append(col)  # Drop original timestamp after extracting features
        
        # Handle other non-feature columns
        for col in ['is_anomaly', 'anomaly_type', 'id', 'label']:
            if col in df.columns:
                drop_cols.append(col)
        
        # Extract target variable if it exists
        if 'is_anomaly' in df.columns:
            y = df['is_anomaly'].values
        else:
            logger.warning("No 'is_anomaly' column found. Creating unlabeled dataset.")
            y = None
        
        # Drop non-feature columns
        X_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Handle categorical features
        cat_columns = X_df.select_dtypes(include=['object', 'category']).columns
        if not cat_columns.empty:
            logger.info(f"One-hot encoding categorical features: {list(cat_columns)}")
            X_df = pd.get_dummies(X_df, columns=cat_columns, drop_first=True)
        
        # Get feature matrix
        X = X_df.values
        
        # Create and fit the scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split the data
        if y is not None:
            # Stratified split to maintain anomaly ratio
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Log split information
            train_anomaly_ratio = np.mean(y_train)
            test_anomaly_ratio = np.mean(y_test)
            logger.info(f"Train set: {X_train.shape[0]} samples, {train_anomaly_ratio:.2%} anomalies")
            logger.info(f"Test set: {X_test.shape[0]} samples, {test_anomaly_ratio:.2%} anomalies")
            
            # Create return dictionary
            data_dict = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler,
                'feature_names': X_df.columns.tolist()
            }
        else:
            # Unlabeled data - use simple train/test split
            X_train, X_test = train_test_split(
                X_scaled, test_size=test_size, random_state=random_state
            )
            logger.info(f"Train set: {X_train.shape[0]} samples (unlabeled)")
            logger.info(f"Test set: {X_test.shape[0]} samples (unlabeled)")
            
            # Create return dictionary
            data_dict = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': None,
                'y_test': None,
                'scaler': scaler,
                'feature_names': X_df.columns.tolist()
            }
        
        # Save the preprocessed data
        logger.info("Saving preprocessed data...")
        
        # Create processed data directory if it doesn't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Save X_train and X_test
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
        
        # Save y_train and y_test if they exist
        if y is not None:
            np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
            np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
        
        # Save the scaler
        joblib.dump(scaler, os.path.join(PROCESSED_DATA_DIR, 'scaler.joblib'))
        
        # Save feature names
        with open(os.path.join(PROCESSED_DATA_DIR, 'feature_names.txt'), 'w') as f:
            for feature in X_df.columns:
                f.write(f"{feature}\n")
        
        logger.info("Preprocessing completed successfully.")
        return data_dict
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None


def main():
    """Main function to execute the data preprocessing pipeline."""
    logger.info("Starting data preprocessing pipeline...")
    
    # Download or create the dataset
    dataset_path = download_dataset()
    
    if dataset_path:
        # Load and explore the data
        df = load_and_explore_data(dataset_path)
        
        if df is not None:
            # Preprocess the data
            data_dict = preprocess_data(df)
            
            if data_dict:
                logger.info("Data preprocessing pipeline completed successfully.")
                return data_dict
    
    logger.error("Data preprocessing pipeline failed.")
    return None


if __name__ == "__main__":
    main() 