#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - App Features
----------------------------------------
This module contains features and components for the Streamlit application.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import base64
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import custom modules
from utils.data_utils import preprocess_data
from utils.model_utils import get_anomaly_scores, evaluate_binary_classification
from utils.visualization_utils import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve


def get_file_download_link(df, filename, text):
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def display_metrics(metrics):
    """Display classification metrics in a nice format."""
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    
    with col4:
        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    
    # Convert the confusion matrix to a DataFrame for better display
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'], 
        index=['Actual Normal', 'Actual Anomaly'],
        columns=['Predicted Normal', 'Predicted Anomaly']
    )
    
    # Display as a table
    st.table(cm_df)


def dataset_exploration_tab(df, dataset_name):
    """
    Dataset exploration tab content
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to explore
    dataset_name : str
        Name of the dataset
    """
    st.header("Dataset Exploration")
    
    # Display basic info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dataset:** {dataset_name}")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
    
    with col2:
        # Check if there's a target column
        if 'label' in df.columns:
            normal_count = (df['label'] == 0).sum()
            anomaly_count = (df['label'] == 1).sum()
            st.write(f"**Normal samples:** {normal_count} ({normal_count/len(df)*100:.2f}%)")
            st.write(f"**Anomaly samples:** {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")
            
            # Create a pie chart
            fig = px.pie(
                values=[normal_count, anomaly_count],
                names=['Normal', 'Anomaly'],
                title='Class Distribution',
                color_discrete_sequence=['#3498db', '#e74c3c'],
                hole=0.4
            )
            st.plotly_chart(fig)
    
    # Display the dataframe
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))
    
    # Display statistics
    st.subheader("Data Statistics")
    st.write(df.describe())
    
    # Data visualization
    st.subheader("Data Visualization")
    
    # Select visualization type
    viz_type = st.selectbox(
        "Select visualization type",
        ["Correlation Heatmap", "Feature Distribution", "PCA Visualization", "Column Data Types"]
    )
    
    if viz_type == "Correlation Heatmap":
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        
        # Plot correlation heatmap using plotly
        fig = px.imshow(
            corr.values,
            x=corr.columns,
            y=corr.columns,
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix',
            labels=dict(color="Correlation")
        )
        fig.update_layout(width=800, height=800)
        st.plotly_chart(fig)
    
    elif viz_type == "Feature Distribution":
        # Select a feature to visualize
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature = st.selectbox("Select feature", numeric_cols)
        
        if 'label' in df.columns:
            # Create distribution plot with plotly
            fig = px.histogram(
                df, 
                x=feature, 
                color='label',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                barmode='overlay',
                marginal='violin',
                labels={'label': 'Class', 0: 'Normal', 1: 'Anomaly'},
                title=f'Distribution of {feature} by Class'
            )
            fig.update_layout(xaxis_title=feature, yaxis_title='Count')
            st.plotly_chart(fig)
        else:
            # Simple distribution without labels
            fig = px.histogram(
                df, 
                x=feature, 
                title=f'Distribution of {feature}',
                marginal='box'
            )
            st.plotly_chart(fig)
    
    elif viz_type == "PCA Visualization":
        # Preprocess data for PCA
        numeric_df = df.select_dtypes(include=['number'])
        
        # Remove label column if it exists
        if 'label' in numeric_df.columns:
            y = numeric_df['label']
            X = numeric_df.drop('label', axis=1)
        else:
            y = None
            X = numeric_df
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA or t-SNE based on user selection
        dim_reduction = st.radio("Dimensionality Reduction Technique", ["PCA", "t-SNE"])
        
        if dim_reduction == "PCA":
            # Apply PCA
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_scaled)
            explained_var = pca.explained_variance_ratio_
            
            # Create dataframe for plotting
            pca_df = pd.DataFrame({
                'PC1': X_reduced[:, 0],
                'PC2': X_reduced[:, 1]
            })
            
            if y is not None:
                pca_df['Class'] = y
                
                # Create PCA plot with plotly
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2', 
                    color='Class',
                    color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                    labels={'Class': 'Class', 0: 'Normal', 1: 'Anomaly'},
                    title=f'PCA Visualization (Explained Variance: PC1={explained_var[0]:.2f}, PC2={explained_var[1]:.2f})'
                )
            else:
                # Simple PCA plot without labels
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2',
                    title=f'PCA Visualization (Explained Variance: PC1={explained_var[0]:.2f}, PC2={explained_var[1]:.2f})'
                )
            
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig)
            
        else:  # t-SNE
            # Only run t-SNE if the dataset is not too large
            if X_scaled.shape[0] > 5000:
                st.warning("Dataset is too large for t-SNE visualization. Using a sample of 5000 points.")
                indices = np.random.choice(X_scaled.shape[0], 5000, replace=False)
                X_sample = X_scaled[indices]
                y_sample = y[indices] if y is not None else None
            else:
                X_sample = X_scaled
                y_sample = y
            
            # Apply t-SNE
            with st.spinner("Computing t-SNE (this may take a while)..."):
                tsne = TSNE(n_components=2, random_state=42)
                X_reduced = tsne.fit_transform(X_sample)
            
            # Create dataframe for plotting
            tsne_df = pd.DataFrame({
                'Dimension 1': X_reduced[:, 0],
                'Dimension 2': X_reduced[:, 1]
            })
            
            if y_sample is not None:
                tsne_df['Class'] = y_sample
                
                # Create t-SNE plot with plotly
                fig = px.scatter(
                    tsne_df, x='Dimension 1', y='Dimension 2', 
                    color='Class',
                    color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                    labels={'Class': 'Class', 0: 'Normal', 1: 'Anomaly'},
                    title='t-SNE Visualization'
                )
            else:
                # Simple t-SNE plot without labels
                fig = px.scatter(
                    tsne_df, x='Dimension 1', y='Dimension 2',
                    title='t-SNE Visualization'
                )
            
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig)
    
    elif viz_type == "Column Data Types":
        # Display column data types
        col_types = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isna().sum()
        })
        st.dataframe(col_types)
        
        # Count of each data type
        type_counts = df.dtypes.value_counts().reset_index()
        type_counts.columns = ['Data Type', 'Count']
        
        # Create a bar chart
        fig = px.bar(
            type_counts, 
            x='Data Type', 
            y='Count',
            title='Column Data Types',
            color='Data Type'
        )
        st.plotly_chart(fig)


def anomaly_detection_tab(df, dataset_name, models_dir):
    """
    Anomaly detection tab content
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to use
    dataset_name : str
        Name of the dataset
    models_dir : str
        Directory containing trained models
    """
    st.header("Anomaly Detection")
    
    # Get available models, prioritizing improved versions
    available_models = []
    improved_models = []
    standard_models = []
    
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '').replace('_', ' ').title()
                
                # Check if it's an improved model
                if 'improved' in filename.lower() or 'fixed' in filename.lower():
                    improved_models.append(model_name)
                else:
                    standard_models.append(model_name)
    
    # Prioritize improved models
    available_models = improved_models + standard_models
    
    if not available_models:
        st.warning("No trained models found. Please run model_training.py first.")
        return
    
    # Model selection
    selected_model_name = st.selectbox(
        "Select a model",
        available_models
    )
    
    # Test size slider
    test_size = st.slider(
        "Test set size (%)",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    ) / 100
    
    # Anomaly threshold slider
    threshold = st.slider(
        "Anomaly threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Run anomaly detection
    if st.button("Run Anomaly Detection"):
        with st.spinner("Processing data and running anomaly detection..."):
            # Preprocess data
            processed_data = preprocess_data(df, test_size=test_size, random_state=42)
            
            # Convert model name back to filename format
            model_filename = selected_model_name.lower().replace(' ', '_') + '.joblib'
            
            # Load model
            model_path = os.path.join(models_dir, model_filename)
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return
                
            model = joblib.load(model_path)
            
            # Get anomaly scores based on model type
            X_test = processed_data['X_test']
            
            # Special handling for DBSCAN
            if 'dbscan' in model_filename.lower():
                if isinstance(model, dict) and 'model' in model:
                    # Use DBSCAN directly on test data
                    from sklearn.cluster import DBSCAN
                    try:
                        eps = model['model'].get_params()['eps']
                        min_samples = model['model'].get_params()['min_samples']
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        dbscan.fit(X_test)
                        # -1 is outlier (anomaly), convert to 1 for consistency
                        y_pred = np.array([1 if label == -1 else 0 for label in dbscan.labels_])
                        # Create fake scores from cluster labels
                        scores = np.zeros(len(y_pred))
                        scores[y_pred == 1] = 0.75  # Anomaly score for outliers
                        scores[y_pred == 0] = 0.25  # Normal score for inliers
                    except Exception as e:
                        st.error(f"Error using DBSCAN model: {e}")
                        return
                else:
                    st.error("DBSCAN model format not recognized")
                    return
            else:
                # For Isolation Forest, One-Class SVM, LOF
            scores = get_anomaly_scores(model, X_test)
            
            # Normalize scores to [0, 1]
            if scores.min() != scores.max():  # Prevent division by zero
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized_scores = np.zeros_like(scores)
            
            # Predict anomalies
            y_pred = (normalized_scores > threshold).astype(int)
            
            # If we have labels, evaluate the model
            if 'y_test' in processed_data and processed_data['y_test'] is not None:
                y_test = processed_data['y_test']
                
                # Evaluate performance
                metrics = evaluate_binary_classification(y_test, y_pred, normalized_scores)
                
                # Display metrics
                st.subheader("Model Performance")
                display_metrics(metrics)
                
                # ROC curve
                if 'fpr' in metrics and 'tpr' in metrics:
                    st.subheader("ROC Curve")
                    fig = px.line(
                        x=metrics['fpr'], 
                        y=metrics['tpr'],
                        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                        title=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})'
                    )
                    fig.add_shape(
                        type='line', line=dict(dash='dash'),
                        x0=0, x1=1, y0=0, y1=1
                    )
                    st.plotly_chart(fig)
                
                # Score distribution
                st.subheader("Anomaly Score Distribution")
                score_df = pd.DataFrame({
                    'Anomaly Score': normalized_scores,
                    'True Class': ['Normal' if y == 0 else 'Anomaly' for y in y_test],
                    'Predicted Class': ['Normal' if y == 0 else 'Anomaly' for y in y_pred]
                })
                
                fig = px.histogram(
                    score_df, 
                    x='Anomaly Score', 
                    color='True Class',
                    barmode='overlay',
                    marginal='violin',
                    title='Distribution of Anomaly Scores by Class'
                )
                fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                              annotation_text=f"Threshold: {threshold}")
                st.plotly_chart(fig)
            else:
                # No labels, just show the predictions
                st.subheader("Anomaly Detection Results")
                st.write(f"**Total samples:** {len(y_pred)}")
                st.write(f"**Detected anomalies:** {np.sum(y_pred)} ({np.sum(y_pred)/len(y_pred)*100:.2f}%)")
                
                # Score distribution
                st.subheader("Anomaly Score Distribution")
                score_df = pd.DataFrame({
                    'Anomaly Score': normalized_scores,
                    'Predicted Class': ['Normal' if y == 0 else 'Anomaly' for y in y_pred]
                })
                
                fig = px.histogram(
                    score_df, 
                    x='Anomaly Score', 
                    color='Predicted Class',
                    barmode='overlay',
                    marginal='violin',
                    title='Distribution of Anomaly Scores'
                )
                fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                              annotation_text=f"Threshold: {threshold}")
                st.plotly_chart(fig)
            
            # Display results table
            st.subheader("Sample Results")
            
            # Get the original test data
            X_test_indices = processed_data.get('test_indices', np.arange(len(scores)))
            test_df = df.iloc[X_test_indices].copy() if X_test_indices is not None else None
            
            if test_df is not None:
                # Add anomaly scores and predictions
                test_df['Anomaly Score'] = normalized_scores
                test_df['Is Anomaly'] = y_pred
                
                # Sort by anomaly score (descending)
                test_df = test_df.sort_values('Anomaly Score', ascending=False)
                
                # Display results
                st.dataframe(test_df.head(20))
                
                # Provide download link
                st.markdown(get_file_download_link(test_df, 'anomaly_detection_results.csv', 
                                               'Download complete results as CSV'), unsafe_allow_html=True)


def model_comparison_tab(models_dir, visualizations_dir):
    """
    Model comparison tab content
    
    Parameters:
    -----------
    models_dir : str
        Directory containing trained models
    visualizations_dir : str
        Directory to save visualizations
    """
    st.header("Model Comparison")
    
    # Create model results on the fly since we now have improved models
    if os.path.exists(models_dir):
        st.subheader("Available Models")
        
        # Group models into improved and standard
        improved_models = []
        standard_models = []
        
        for filename in os.listdir(models_dir):
            if filename.endswith('.joblib'):
                if 'improved' in filename.lower() or 'fixed' in filename.lower():
                    improved_models.append(filename)
                else:
                    standard_models.append(filename)
        
        # Display model list
        if improved_models:
            st.write("**Improved Models:**")
            for model in improved_models:
                st.write(f"- {model.replace('.joblib', '').replace('_', ' ').title()}")
        
        if standard_models:
            st.write("**Standard Models:**")
            for model in standard_models:
                st.write(f"- {model.replace('.joblib', '').replace('_', ' ').title()}")
    
    # Check if we have model results
    results_csv = os.path.join(visualizations_dir, 'model_metrics.csv')
    
    if os.path.exists(results_csv):
        # Load results
        results_df = pd.read_csv(results_csv)
        
        # Display results table
        st.subheader("Model Performance Metrics")
        st.dataframe(results_df)
        
        # Bar chart comparing key metrics
        st.subheader("Performance Comparison")
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'roc_auc' in results_df.columns:
            metrics.append('roc_auc')
        
        # Let user select metrics to compare
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            metrics,
            default=['accuracy', 'f1_score']
        )
        
        if selected_metrics:
            # Melt the dataframe for easier plotting
            plot_df = results_df.melt(
                id_vars=['model_name'],
                value_vars=selected_metrics,
                var_name='Metric',
                value_name='Value'
            )
            
            # Create bar chart
            fig = px.bar(
                plot_df,
                x='model_name',
                y='Value',
                color='Metric',
                barmode='group',
                title='Model Performance Comparison',
                labels={'model_name': 'Model', 'Value': 'Score'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        
        # Training time comparison
        if 'training_time' in results_df.columns:
            st.subheader("Training Time Comparison")
            fig = px.bar(
                results_df,
                x='model_name',
                y='training_time',
                title='Model Training Time (seconds)',
                labels={'model_name': 'Model', 'training_time': 'Training Time (s)'},
                color='model_name'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        
        # Show best model
        if 'f1_score' in results_df.columns:
            best_model = results_df.loc[results_df['f1_score'].idxmax()]
            
            st.subheader("Best Performing Model")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Best Model:** {best_model['model_name']}")
                st.info(f"**F1 Score:** {best_model['f1_score']:.4f}")
                st.info(f"**Accuracy:** {best_model['accuracy']:.4f}")
            
            with col2:
                if 'precision' in best_model and 'recall' in best_model:
                    st.info(f"**Precision:** {best_model['precision']:.4f}")
                    st.info(f"**Recall:** {best_model['recall']:.4f}")
                if 'roc_auc' in best_model:
                    st.info(f"**ROC AUC:** {best_model['roc_auc']:.4f}")
    
    else:
        # No results found
        st.warning("No model comparison results found. Please run model_training.py first to generate comparison data.")
        
        # Check if we have any models at all
        available_models = []
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.joblib'):
                    model_name = filename.replace('.joblib', '').replace('_', ' ').title()
                    available_models.append(model_name)
        
        if available_models:
            st.subheader("Available Models")
            for model in available_models:
                st.write(f"- {model}")
        else:
            st.error("No trained models found. Please run model_training.py first.")


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
                st.markdown("Go to the **Dataset Exploration** tab to explore your dataset.") 