#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Visualization Utilities
------------------------------------------------
This module contains utility functions for data and results visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualization_utils')


def plot_feature_distributions(df, features=None, hue=None, n_cols=3, figsize=(15, 15)):
    """
    Plot distributions of features in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the features
    features : list or None
        List of features to plot (if None, all numeric features are used)
    hue : str or None
        Column to use for coloring
    n_cols : int
        Number of columns in the grid
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots
    """
    try:
        # Select features to plot
        if features is None:
            # Use all numeric columns
            features = df.select_dtypes(include=['number']).columns.tolist()
            
            # Exclude hue column if it's numeric
            if hue in features:
                features.remove(hue)
        
        if not features:
            logger.warning("No features to plot")
            return None
        
        # Calculate grid dimensions
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        # Plot each feature
        for i, feature in enumerate(features):
            ax = axes[i]
            
            if hue is not None and hue in df.columns:
                # Plot by hue
                for hue_value in sorted(df[hue].unique()):
                    subset = df[df[hue] == hue_value]
                    sns.kdeplot(subset[feature], ax=ax, label=f'{hue}={hue_value}')
                
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
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {e}")
        return None


def plot_correlation_heatmap(df, features=None, figsize=(12, 10), cmap='coolwarm', mask_upper=True):
    """
    Plot correlation heatmap of features in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the features
    features : list or None
        List of features to include (if None, all numeric features are used)
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the heatmap
    mask_upper : bool
        Whether to mask the upper triangle
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the heatmap
    """
    try:
        # Select features to include
        if features is None:
            # Use all numeric columns
            features = df.select_dtypes(include=['number']).columns.tolist()
        
        if not features:
            logger.warning("No features to plot")
            return None
        
        # Select DataFrame with only the desired features
        plot_df = df[features]
        
        # Compute correlation matrix
        corr = plot_df.corr()
        
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
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {e}")
        return None


def plot_pca_components(X, y=None, n_components=2, figsize=(10, 8), random_state=42):
    """
    Plot PCA components of the data.
    
    Parameters:
    -----------
    X : numpy.ndarray or pd.DataFrame
        Data to transform
    y : numpy.ndarray or None
        Labels for coloring (if None, no coloring is used)
    n_components : int
        Number of components to compute
    figsize : tuple
        Figure size
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        X_pca = pca.fit_transform(X)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        if n_components == 2:
            # 2D plot
            if y is not None:
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolors='k', linewidth=0.5)
                plt.colorbar(label='Class')
            else:
                plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8, edgecolors='k', linewidth=0.5)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA: First Two Principal Components')
        
        elif n_components == 3:
            # 3D plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            if y is not None:
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.8, edgecolors='k', linewidth=0.5)
                plt.colorbar(scatter, label='Class')
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.8, edgecolors='k', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
            plt.title('PCA: First Three Principal Components')
        
        else:
            # For more than 3 components, show explained variance
            plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA: Explained Variance by Component')
            plt.xticks(range(1, n_components + 1))
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting PCA components: {e}")
        return None


def plot_tsne(X, y=None, perplexity=30, n_components=2, figsize=(10, 8), random_state=42):
    """
    Plot t-SNE visualization of the data.
    
    Parameters:
    -----------
    X : numpy.ndarray or pd.DataFrame
        Data to transform
    y : numpy.ndarray or None
        Labels for coloring (if None, no coloring is used)
    perplexity : float
        Perplexity parameter for t-SNE
    n_components : int
        Number of components to compute
    figsize : tuple
        Figure size
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Check if we have too many samples for t-SNE
        max_samples = 5000
        if X.shape[0] > max_samples:
            logger.warning(f"Too many samples for t-SNE ({X.shape[0]} > {max_samples}). Subsampling...")
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[indices]
            if y is not None:
                y_sample = y[indices]
            else:
                y_sample = None
        else:
            X_sample = X
            y_sample = y
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        X_tsne = tsne.fit_transform(X_sample)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        if n_components == 2:
            # 2D plot
            if y_sample is not None:
                plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.8, edgecolors='k', linewidth=0.5)
                plt.colorbar(label='Class')
            else:
                plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.8, edgecolors='k', linewidth=0.5)
            
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.title('t-SNE Visualization')
        
        elif n_components == 3:
            # 3D plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            if y_sample is not None:
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y_sample, cmap='viridis', alpha=0.8, edgecolors='k', linewidth=0.5)
                plt.colorbar(scatter, label='Class')
            else:
                ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], alpha=0.8, edgecolors='k', linewidth=0.5)
            
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')
            plt.title('t-SNE Visualization')
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting t-SNE: {e}")
        return None


def plot_anomaly_scores(scores, threshold=None, labels=None, figsize=(12, 6)):
    """
    Plot anomaly scores.
    
    Parameters:
    -----------
    scores : numpy.ndarray
        Anomaly scores
    threshold : float or None
        Threshold for anomaly detection (if None, no threshold is shown)
    labels : numpy.ndarray or None
        True labels (0: normal, 1: anomaly) for coloring points
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot scores
        x = np.arange(len(scores))
        
        if labels is not None:
            # Color by true class
            normal_indices = np.where(labels == 0)[0]
            anomaly_indices = np.where(labels == 1)[0]
            
            plt.scatter(normal_indices, scores[normal_indices], color='blue', alpha=0.6, label='Normal')
            plt.scatter(anomaly_indices, scores[anomaly_indices], color='red', alpha=0.6, label='Anomaly')
            plt.legend()
        else:
            # No true labels
            plt.scatter(x, scores, alpha=0.6)
        
        # Add threshold line if provided
        if threshold is not None:
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
            plt.legend()
        
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting anomaly scores: {e}")
        return None


def plot_feature_importance(feature_names, importance_values, top_n=20, figsize=(10, 8)):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_names : list
        Names of the features
    importance_values : numpy.ndarray
        Importance values for each feature
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Select top N features
        if top_n is not None and top_n < len(importance_df):
            importance_df = importance_df.head(top_n)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        return None


def plot_model_comparison(model_names, metrics, metric_names=None, figsize=(12, 8)):
    """
    Plot comparison of model performance.
    
    Parameters:
    -----------
    model_names : list
        Names of the models
    metrics : numpy.ndarray or list of lists
        Metrics for each model (shape: n_models x n_metrics)
    metric_names : list or None
        Names of the metrics (if None, generic names are used)
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Convert metrics to numpy array
        metrics_array = np.array(metrics)
        
        # Check dimensions
        n_models = len(model_names)
        n_metrics = metrics_array.shape[1]
        
        # Set default metric names if not provided
        if metric_names is None:
            metric_names = [f'Metric {i+1}' for i in range(n_metrics)]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Set up bar positions
        x = np.arange(n_models)
        width = 0.8 / n_metrics
        
        # Plot bars for each metric
        for i, metric_name in enumerate(metric_names):
            plt.bar(x + (i - n_metrics/2 + 0.5) * width, metrics_array[:, i], 
                   width, label=metric_name)
        
        # Set plot properties
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting model comparison: {e}")
        return None


def plot_confusion_matrices(confusion_matrices, model_names, figsize=(15, 10)):
    """
    Plot confusion matrices for multiple models.
    
    Parameters:
    -----------
    confusion_matrices : list
        List of confusion matrices
    model_names : list
        Names of the models
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots
    """
    try:
        # Number of models
        n_models = len(model_names)
        
        # Calculate grid dimensions
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        # Plot each confusion matrix
        for i, (cm, model_name) in enumerate(zip(confusion_matrices, model_names)):
            ax = axes[i]
            
            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Calculate accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            
            # Plot heatmap
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            
            ax.set_title(f"{model_name}\nAccuracy: {accuracy:.4f}")
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrices: {e}")
        return None


def plot_roc_curves(fpr_list, tpr_list, roc_auc_list, model_names, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models.
    
    Parameters:
    -----------
    fpr_list : list
        List of false positive rates for each model
    tpr_list : list
        List of true positive rates for each model
    roc_auc_list : list
        List of ROC AUC values for each model
    model_names : list
        Names of the models
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot ROC curve for each model
        for fpr, tpr, roc_auc, model_name in zip(fpr_list, tpr_list, roc_auc_list, model_names):
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot random guess line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error plotting ROC curves: {e}")
        return None


def create_interactive_tsne(X, y=None, anomaly_scores=None, perplexity=30, random_state=42):
    """
    Create an interactive t-SNE visualization using Plotly.
    
    Parameters:
    -----------
    X : numpy.ndarray or pd.DataFrame
        Data to transform
    y : numpy.ndarray or None
        Labels for coloring (if None, no coloring is used)
    anomaly_scores : numpy.ndarray or None
        Anomaly scores for coloring (if None, y is used for coloring)
    perplexity : float
        Perplexity parameter for t-SNE
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    try:
        # Check if we have too many samples for t-SNE
        max_samples = 5000
        if X.shape[0] > max_samples:
            logger.warning(f"Too many samples for t-SNE ({X.shape[0]} > {max_samples}). Subsampling...")
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[indices]
            if y is not None:
                y_sample = y[indices]
            else:
                y_sample = None
            if anomaly_scores is not None:
                anomaly_scores_sample = anomaly_scores[indices]
            else:
                anomaly_scores_sample = None
        else:
            X_sample = X
            y_sample = y
            anomaly_scores_sample = anomaly_scores
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        X_tsne = tsne.fit_transform(X_sample)
        
        # Create DataFrame for Plotly
        df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
        
        # Add labels and/or anomaly scores if available
        if y_sample is not None:
            df_tsne['Label'] = y_sample
            df_tsne['Label'] = df_tsne['Label'].map({0: 'Normal', 1: 'Anomaly'})
        
        if anomaly_scores_sample is not None:
            df_tsne['Anomaly Score'] = anomaly_scores_sample
        
        # Create Plotly figure
        if anomaly_scores_sample is not None:
            # Color by anomaly score
            fig = px.scatter(
                df_tsne, x='TSNE1', y='TSNE2', color='Anomaly Score',
                color_continuous_scale='Viridis',
                hover_data=['Anomaly Score'],
                title='t-SNE Visualization of Anomaly Scores'
            )
            
            # Add true labels as marker symbols if available
            if y_sample is not None:
                fig.update_traces(marker=dict(symbol=df_tsne['Label'].map({'Normal': 'circle', 'Anomaly': 'x'})))
        
        elif y_sample is not None:
            # Color by true label
            fig = px.scatter(
                df_tsne, x='TSNE1', y='TSNE2', color='Label',
                color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
                title='t-SNE Visualization of True Labels'
            )
        
        else:
            # No coloring
            fig = px.scatter(
                df_tsne, x='TSNE1', y='TSNE2',
                title='t-SNE Visualization'
            )
        
        # Update layout
        fig.update_layout(
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            legend_title='',
            height=600,
            width=800
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating interactive t-SNE: {e}")
        return None


def create_3d_pca(X, y=None, anomaly_scores=None, random_state=42):
    """
    Create an interactive 3D PCA visualization using Plotly.
    
    Parameters:
    -----------
    X : numpy.ndarray or pd.DataFrame
        Data to transform
    y : numpy.ndarray or None
        Labels for coloring (if None, no coloring is used)
    anomaly_scores : numpy.ndarray or None
        Anomaly scores for coloring (if None, y is used for coloring)
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    try:
        # Apply PCA
        pca = PCA(n_components=3, random_state=random_state)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame for Plotly
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
        
        # Add variance explained
        var_explained = pca.explained_variance_ratio_ * 100
        
        # Add labels and/or anomaly scores if available
        if y is not None:
            df_pca['Label'] = y
            df_pca['Label'] = df_pca['Label'].map({0: 'Normal', 1: 'Anomaly'})
        
        if anomaly_scores is not None:
            df_pca['Anomaly Score'] = anomaly_scores
        
        # Create Plotly figure
        if anomaly_scores is not None:
            # Color by anomaly score
            fig = px.scatter_3d(
                df_pca, x='PC1', y='PC2', z='PC3', color='Anomaly Score',
                color_continuous_scale='Viridis',
                hover_data=['Anomaly Score'],
                title='3D PCA Visualization of Anomaly Scores'
            )
            
            # Add true labels as marker symbols if available
            if y is not None:
                fig.update_traces(marker=dict(symbol=df_pca['Label'].map({'Normal': 'circle', 'Anomaly': 'x'})))
        
        elif y is not None:
            # Color by true label
            fig = px.scatter_3d(
                df_pca, x='PC1', y='PC2', z='PC3', color='Label',
                color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
                title='3D PCA Visualization of True Labels'
            )
        
        else:
            # No coloring
            fig = px.scatter_3d(
                df_pca, x='PC1', y='PC2', z='PC3',
                title='3D PCA Visualization'
            )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({var_explained[0]:.2f}% variance)',
                yaxis_title=f'PC2 ({var_explained[1]:.2f}% variance)',
                zaxis_title=f'PC3 ({var_explained[2]:.2f}% variance)'
            ),
            height=700,
            width=900
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating 3D PCA: {e}")
        return None


def create_anomaly_dashboard(anomaly_scores, threshold, true_labels=None, feature_values=None, feature_names=None):
    """
    Create an interactive dashboard for anomaly detection using Plotly.
    
    Parameters:
    -----------
    anomaly_scores : numpy.ndarray
        Anomaly scores
    threshold : float
        Threshold for anomaly detection
    true_labels : numpy.ndarray or None
        True labels (0: normal, 1: anomaly) if available
    feature_values : numpy.ndarray or None
        Feature values for displaying details
    feature_names : list or None
        Names of the features
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    try:
        # Create binary predictions
        predictions = (anomaly_scores > threshold).astype(int)
        
        # Create data for plotting
        df_plot = pd.DataFrame({
            'Sample': np.arange(len(anomaly_scores)),
            'Anomaly Score': anomaly_scores,
            'Prediction': predictions.map({0: 'Normal', 1: 'Anomaly'})
        })
        
        # Add true labels if available
        if true_labels is not None:
            df_plot['True Label'] = true_labels.map({0: 'Normal', 1: 'Anomaly'})
        
        # Create subplots
        if true_labels is not None:
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"colspan": 2}, None],
                    [{"type": "pie"}, {"type": "pie"}]
                ],
                subplot_titles=(
                    'Anomaly Scores',
                    'Prediction Distribution',
                    'True Label Distribution'
                ),
                vertical_spacing=0.15
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                specs=[
                    [{}],
                    [{"type": "pie"}]
                ],
                subplot_titles=(
                    'Anomaly Scores',
                    'Prediction Distribution'
                ),
                vertical_spacing=0.15
            )
        
        # Add anomaly scores scatter plot
        fig.add_trace(
            go.Scatter(
                x=df_plot['Sample'],
                y=df_plot['Anomaly Score'],
                mode='markers',
                marker=dict(
                    color=df_plot['Prediction'].map({'Normal': 'blue', 'Anomaly': 'red'}),
                    size=8
                ),
                name='Anomaly Score'
            ),
            row=1, col=1
        )
        
        # Add threshold line
        fig.add_trace(
            go.Scatter(
                x=[0, len(anomaly_scores) - 1],
                y=[threshold, threshold],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Threshold'
            ),
            row=1, col=1
        )
        
        # Add prediction distribution pie chart
        prediction_counts = df_plot['Prediction'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=prediction_counts.index,
                values=prediction_counts.values,
                marker=dict(colors=['blue', 'red']),
                name='Predictions'
            ),
            row=2, col=1 if true_labels is None else 1
        )
        
        # Add true label distribution pie chart if available
        if true_labels is not None:
            true_label_counts = df_plot['True Label'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=true_label_counts.index,
                    values=true_label_counts.values,
                    marker=dict(colors=['blue', 'red']),
                    name='True Labels'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Anomaly Detection Dashboard',
            height=800,
            width=1000,
            showlegend=True
        )
        
        # Update xaxis and yaxis properties
        fig.update_xaxes(title_text='Sample Index', row=1, col=1)
        fig.update_yaxes(title_text='Anomaly Score', row=1, col=1)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating anomaly dashboard: {e}")
        return None


if __name__ == "__main__":
    # Example usage    pass 
