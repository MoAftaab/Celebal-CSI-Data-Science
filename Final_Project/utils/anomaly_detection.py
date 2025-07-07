#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Anomaly Detection Features
-----------------------------------------------------
This module contains features for anomaly detection and model comparison.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Import custom modules
from utils.data_utils import preprocess_data
from utils.model_utils import get_anomaly_scores, evaluate_binary_classification
from utils.app_features import get_file_download_link, display_metrics


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
    
    # Get available models
    available_models = []
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '').replace('_', ' ').title()
                available_models.append(model_name)
    
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
            
            # Load model
            model_path = os.path.join(models_dir, f"{selected_model_name.lower().replace(' ', '_')}.joblib")
            model = joblib.load(model_path)
            
            # Get anomaly scores
            X_test = processed_data['X_test']
            scores = get_anomaly_scores(model, X_test)
            
            # Normalize scores to [0, 1]
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            
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
    
    # Check if we have model results
    results_csv = os.path.join(visualizations_dir, 'model_results.csv')
    
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