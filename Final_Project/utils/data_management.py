#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Data Management
------------------------------------------
This module contains functions for data upload and management.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


def upload_data_tab(datasets_dir):
    """
    Upload data tab content
    
    Parameters:
    -----------
    datasets_dir : str
        Directory to save uploaded datasets
    """
    st.header("Upload Data")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df.head(5))
            
            # Basic info
            st.subheader("Basic Information")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            
            # Save the file
            save_name = st.text_input("Save as", uploaded_file.name)
            
            if st.button("Save Dataset"):
                # Create directory if it doesn't exist
                os.makedirs(datasets_dir, exist_ok=True)
                
                # Save the file
                save_path = os.path.join(datasets_dir, save_name)
                df.to_csv(save_path, index=False)
                st.success(f"Dataset saved as {save_name}")
                
                # Provide a link to explore the dataset
                st.markdown("Go to the **Dataset Exploration** tab to explore your dataset.")
        
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    
    # Data generation option
    st.subheader("Or Generate Synthetic Data")
    
    with st.expander("Generate Synthetic Network Traffic Data"):
        # Parameters for synthetic data
        n_samples = st.slider("Number of samples", 100, 10000, 1000)
        anomaly_percentage = st.slider("Anomaly percentage", 1, 30, 10)
        n_features = st.slider("Number of features", 5, 30, 10)
        
        if st.button("Generate Data"):
            # Create synthetic data
            np.random.seed(42)
            
            # Generate normal samples
            n_normal = int(n_samples * (1 - anomaly_percentage/100))
            n_anomaly = n_samples - n_normal
            
            # Generate feature names
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # Generate normal samples
            X_normal = np.random.randn(n_normal, n_features)
            
            # Generate anomaly samples with different distribution
            X_anomaly = np.random.randn(n_anomaly, n_features) * 3 + 2
            
            # Combine samples
            X = np.vstack([X_normal, X_anomaly])
            y = np.zeros(n_samples)
            y[n_normal:] = 1
            
            # Create DataFrame
            synthetic_df = pd.DataFrame(X, columns=feature_names)
            synthetic_df['label'] = y
            
            # Display preview
            st.subheader("Generated Data Preview")
            st.dataframe(synthetic_df.head())
            
            # Save the file
            save_name = st.text_input("Save generated data as", "synthetic_network_data.csv")
            
            if st.button("Save Generated Data"):
                # Create directory if it doesn't exist
                os.makedirs(datasets_dir, exist_ok=True)
                
                # Save the file
                save_path = os.path.join(datasets_dir, save_name)
                synthetic_df.to_csv(save_path, index=False)
                st.success(f"Dataset saved as {save_name}")
                
                # Provide a link to explore the dataset
                st.markdown("Go to the **Dataset Exploration** tab to explore your dataset.")


def get_available_datasets(datasets_dir):
    """
    Get list of available datasets.
    
    Parameters:
    -----------
    datasets_dir : str
        Directory containing datasets
    
    Returns:
    --------
    list
        List of dataset filenames
    """
    datasets = []
    
    # Check if datasets directory exists
    if os.path.exists(datasets_dir):
        # List CSV files in the datasets directory
        for filename in os.listdir(datasets_dir):
            if filename.endswith('.csv'):
                datasets.append(filename)
    
    return datasets


def get_available_models(models_dir):
    """
    Get list of available trained models.
    
    Parameters:
    -----------
    models_dir : str
        Directory containing trained models
    
    Returns:
    --------
    list
        List of model names
    """
    models = []
    
    # Check if models directory exists
    if os.path.exists(models_dir):
        # List joblib files in the models directory
        for filename in os.listdir(models_dir):
            if filename.endswith('.joblib'):
                # Convert filename to model name
                model_name = filename.replace('.joblib', '').replace('_', ' ').title()
                models.append(model_name)
    
    return models 