#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Data Utilities
----------------------------------------
This module contains utility functions for data processing and visualization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_utils')


def load_dataset(file_path):
    """
    Load a dataset from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset file
    
    Returns:
    --------
    pd.DataFrame
        The loaded dataset
    """
    try:
        # Check file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
        
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None


def save_dataset(df, file_path):
    """
    Save a dataset to a file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to save
    file_path : str
        Path where to save the dataset
    
    Returns:
    --------
    bool
        True if the dataset was saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check file extension
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif file_path.endswith('.parquet'):
            df.to_parquet(file_path, index=False)
        elif file_path.endswith('.json'):
            df.to_json(file_path, orient='records')
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return False
        
        logger.info(f"Dataset saved successfully to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return False


def scale_features(X, scaler_type='standard', feature_range=(0, 1)):
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Parameters:
    -----------
    X : numpy.ndarray or pd.DataFrame
        The features to scale
    scaler_type : str
        Type of scaler to use ('standard' or 'minmax')
    feature_range : tuple
        Range of the scaled features (for MinMaxScaler)
    
    Returns:
    --------
    tuple
        (scaled_X, scaler)
    """
    try:
        # Convert to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Create and fit the scaler
        if scaler_type.lower() == 'standard':
            scaler = StandardScaler()
        elif scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        else:
            logger.error(f"Unsupported scaler type: {scaler_type}")
            return None, None
        
        X_scaled = scaler.fit_transform(X_array)
        
        logger.info(f"Features scaled successfully using {scaler_type} scaler")
        return X_scaled, scaler
    
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        return None, None


def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame with missing values
    strategy : str
        Strategy to handle missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_value')
    fill_value : float or dict
        Value to use for filling missing values if strategy is 'fill_value'
        Can be a single value or a dictionary mapping column names to fill values
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
    try:
        # Create a copy of the DataFrame
        df_clean = df.copy()
        
        # Check missing values
        missing_values = df_clean.isnull().sum()
        missing_columns = missing_values[missing_values > 0].index.tolist()
        
        if not missing_columns:
            logger.info("No missing values to handle")
            return df_clean
        
        logger.info(f"Handling missing values in columns: {missing_columns}")
        
        # Handle missing values based on strategy
        if strategy == 'drop':
            df_clean = df_clean.dropna()
            logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
        
        elif strategy == 'fill_mean':
            for col in missing_columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    logger.warning(f"Column {col} is not numeric. Cannot fill with mean.")
        
        elif strategy == 'fill_median':
            for col in missing_columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    logger.warning(f"Column {col} is not numeric. Cannot fill with median.")
        
        elif strategy == 'fill_mode':
            for col in missing_columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        elif strategy == 'fill_value':
            if isinstance(fill_value, dict):
                # Fill each column with its specified value
                for col, val in fill_value.items():
                    if col in df_clean.columns:
                        df_clean[col] = df_clean[col].fillna(val)
                    else:
                        logger.warning(f"Column {col} not found in DataFrame")
            else:
                # Fill all columns with the same value
                df_clean = df_clean.fillna(fill_value)
        
        else:
            logger.error(f"Unsupported strategy: {strategy}")
            return df
        
        logger.info("Missing values handled successfully")
        return df_clean
    
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        return df


def detect_outliers(X, method='zscore', threshold=3):
    """
    Detect outliers in a dataset.
    
    Parameters:
    -----------
    X : numpy.ndarray or pd.DataFrame
        The dataset to check for outliers
    method : str
        Method to use for outlier detection ('zscore', 'iqr')
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    numpy.ndarray
        Boolean mask where True indicates an outlier
    """
    try:
        # Convert to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Initialize outlier mask
        outliers_mask = np.zeros(X_array.shape[0], dtype=bool)
        
        # Detect outliers based on method
        if method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(X_array, axis=0, nan_policy='omit'))
            outliers_mask = np.any(z_scores > threshold, axis=1)
        
        elif method == 'iqr':
            # IQR method
            q1 = np.percentile(X_array, 25, axis=0)
            q3 = np.percentile(X_array, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i in range(X_array.shape[1]):
                col_outliers = (X_array[:, i] < lower_bound[i]) | (X_array[:, i] > upper_bound[i])
                outliers_mask = outliers_mask | col_outliers
        
        else:
            logger.error(f"Unsupported method: {method}")
            return None
        
        logger.info(f"Detected {np.sum(outliers_mask)} outliers out of {X_array.shape[0]} samples")
        return outliers_mask
    
    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        return None


def plot_feature_distributions(df, target_col=None, n_cols=3, figsize=(15, 15)):
    """
    Plot distributions of features in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the features
    target_col : str or None
        Name of the target column for coloring
    n_cols : int
        Number of columns in the grid of plots
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots
    """
    try:
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if not numeric_cols:
            logger.warning("No numeric columns found in DataFrame")
            return None
        
        # Calculate grid dimensions
        n_features = len(numeric_cols)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Plot each feature
        for i, feature in enumerate(numeric_cols):
            ax = axes[i]
            
            if target_col is not None and target_col in df.columns:
                # Plot distribution by target
                for target_value in df[target_col].unique():
                    subset = df[df[target_col] == target_value]
                    sns.kdeplot(subset[feature], ax=ax, label=f'{target_col}={target_value}')
                
                ax.legend()
            else:
                # Simple distribution
                sns.histplot(df[feature], ax=ax, kde=True)
            
            ax.set_title(feature)
            ax.set_xlabel('')
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        logger.info(f"Created distribution plots for {n_features} features")
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {e}")
        return None


def plot_correlation_matrix(df, figsize=(12, 10), cmap='coolwarm', mask_upper=True):
    """
    Plot correlation matrix of features in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the features
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the heatmap
    mask_upper : bool
        Whether to mask the upper triangle of the correlation matrix
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the correlation matrix plot
    """
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            logger.warning("No numeric columns found in DataFrame")
            return None
        
        # Compute correlation matrix
        corr = numeric_df.corr()
        
        # Create mask for upper triangle if requested
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        else:
            mask = None
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt='.2f', square=True, linewidths=.5)
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        logger.info(f"Created correlation matrix plot for {len(corr)} features")
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {e}")
        return None


def create_time_features(df, timestamp_col):
    """
    Create time-based features from a timestamp column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the timestamp column
    timestamp_col : str
        Name of the timestamp column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added time features
    """
    try:
        # Create a copy of the DataFrame
        df_new = df.copy()
        
        # Check if the timestamp column exists
        if timestamp_col not in df_new.columns:
            logger.error(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            return df_new
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_new[timestamp_col]):
            df_new[timestamp_col] = pd.to_datetime(df_new[timestamp_col], errors='coerce')
        
        # Check for conversion errors
        if df_new[timestamp_col].isnull().sum() > 0:
            logger.warning(f"Some values in '{timestamp_col}' could not be converted to datetime")
        
        # Extract time features
        df_new[f'{timestamp_col}_hour'] = df_new[timestamp_col].dt.hour
        df_new[f'{timestamp_col}_day'] = df_new[timestamp_col].dt.day
        df_new[f'{timestamp_col}_weekday'] = df_new[timestamp_col].dt.weekday
        df_new[f'{timestamp_col}_month'] = df_new[timestamp_col].dt.month
        df_new[f'{timestamp_col}_year'] = df_new[timestamp_col].dt.year
        df_new[f'{timestamp_col}_quarter'] = df_new[timestamp_col].dt.quarter
        
        # Create cyclical features for hour, weekday, and month
        df_new[f'{timestamp_col}_hour_sin'] = np.sin(2 * np.pi * df_new[f'{timestamp_col}_hour'] / 24)
        df_new[f'{timestamp_col}_hour_cos'] = np.cos(2 * np.pi * df_new[f'{timestamp_col}_hour'] / 24)
        
        df_new[f'{timestamp_col}_weekday_sin'] = np.sin(2 * np.pi * df_new[f'{timestamp_col}_weekday'] / 7)
        df_new[f'{timestamp_col}_weekday_cos'] = np.cos(2 * np.pi * df_new[f'{timestamp_col}_weekday'] / 7)
        
        df_new[f'{timestamp_col}_month_sin'] = np.sin(2 * np.pi * df_new[f'{timestamp_col}_month'] / 12)
        df_new[f'{timestamp_col}_month_cos'] = np.cos(2 * np.pi * df_new[f'{timestamp_col}_month'] / 12)
        
        logger.info(f"Created time features from '{timestamp_col}'")
        return df_new
    
    except Exception as e:
        logger.error(f"Error creating time features: {e}")
        return df


def encode_categorical_features(df, method='onehot', drop_first=True, max_categories=10):
    """
    Encode categorical features in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing categorical features
    method : str
        Encoding method ('onehot', 'label', 'ordinal')
    drop_first : bool
        Whether to drop the first category in one-hot encoding
    max_categories : int
        Maximum number of categories to encode (for one-hot encoding)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical features
    """
    try:
        # Create a copy of the DataFrame
        df_encoded = df.copy()
        
        # Get categorical columns
        cat_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_columns:
            logger.info("No categorical columns found in DataFrame")
            return df_encoded
        
        logger.info(f"Encoding {len(cat_columns)} categorical features using {method} encoding")
        
        if method == 'onehot':
            # One-hot encoding
            for col in cat_columns:
                # Check number of unique values
                unique_values = df_encoded[col].nunique()
                
                if unique_values > max_categories:
                    logger.warning(f"Column '{col}' has {unique_values} categories, which exceeds the maximum of {max_categories}. Skipping.")
                    continue
                
                # Get dummies
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
                
                # Add to DataFrame
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Drop original column
                df_encoded = df_encoded.drop(col, axis=1)
        
        elif method == 'label':
            # Label encoding
            from sklearn.preprocessing import LabelEncoder
            
            for col in cat_columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        elif method == 'ordinal':
            # Ordinal encoding (simple numeric mapping)
            for col in cat_columns:
                categories = df_encoded[col].unique()
                mapping = {category: i for i, category in enumerate(categories)}
                df_encoded[col] = df_encoded[col].map(mapping)
        
        else:
            logger.error(f"Unsupported encoding method: {method}")
            return df
        
        logger.info("Categorical features encoded successfully")
        return df_encoded
    
    except Exception as e:
        logger.error(f"Error encoding categorical features: {e}")
        return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data for anomaly detection.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to preprocess
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
        from sklearn.model_selection import train_test_split
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Separate features and target (if available)
        if 'label' in processed_df.columns:
            y = processed_df['label'].values
            X_df = processed_df.drop('label', axis=1)
        else:
            y = None
            X_df = processed_df
        
        # Handle non-numeric columns
        numeric_cols = X_df.select_dtypes(include=['number']).columns
        non_numeric_cols = X_df.select_dtypes(exclude=['number']).columns
        
        # For each non-numeric column, convert to numeric representation
        for col in non_numeric_cols:
            try:
                # Hash the string values
                X_df[f"{col}_hash"] = pd.util.hash_array(X_df[col].values)
                X_df = X_df.drop(col, axis=1)
                
            except Exception as e:
                logger.warning(f"Could not process column {col}: {e}")
                # If we can't process it, drop the column
                X_df = X_df.drop(col, axis=1)
        
        # Convert to numpy array
        X = X_df.values
        
        # Keep track of original indices
        indices = np.arange(len(X))
        
        # Split into train and test sets
        if y is not None:
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                X, y, indices, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            # If no labels, just split the data
            X_train, X_test, train_indices, test_indices = train_test_split(
                X, indices, test_size=test_size, random_state=random_state
            )
            y_train, y_test = None, None
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create a dictionary with processed data
        data_dict = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'scaler': scaler,
            'feature_names': X_df.columns.tolist()
        }
        
        logger.info(f"Data preprocessed successfully. X_train shape: {X_train_scaled.shape}")
        return data_dict
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    pass 
 