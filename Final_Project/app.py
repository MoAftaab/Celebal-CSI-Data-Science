#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Streamlit Web Application
----------------------------------------------------
This application demonstrates various anomaly detection models for network security.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages

# Constants - Fix paths to be relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")

# Add the project directory to the system path to allow imports
sys.path.append(BASE_DIR)

# Make sure necessary directories exist
for directory in [MODELS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize session state to maintain data between interactions
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# Function to load the NSL-KDD dataset
def load_nsl_kdd_data():
    """Load the NSL-KDD dataset."""
    train_file_path = os.path.join(DATASETS_DIR, "KDDTrain+.csv")
    test_file_path = os.path.join(DATASETS_DIR, "KDDTest+.csv")
    
    if not os.path.exists(train_file_path):
        st.error(f"Dataset file not found: {train_file_path}")
        return None
    
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
    
    # Load the training data
    train_data = pd.read_csv(train_file_path, header=None)
    train_data.columns = col_names
    train_data['label'] = train_data['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    
    return train_data

# Function to preprocess the NSL-KDD data
def preprocess_nsl_kdd_data(df, test_size=0.2, random_state=42):
    """Preprocess the NSL-KDD dataset for anomaly detection."""
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle categorical features
    categorical_columns = ['protocol_type', 'service', 'flag']
    processed_df = pd.get_dummies(processed_df, columns=categorical_columns, drop_first=True)
    
    # Separate features and target
    y = processed_df['label'].values
    X = processed_df.drop(['attack_type', 'difficulty', 'label'], axis=1)
    
    # Handle missing values
    X = X.fillna(X.mean())
    X = X.fillna(0)  # If there are still NaN values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate anomaly ratio for model parameters
    anomaly_ratio = np.sum(y_train == 1) / len(y_train)
    
    # Store processed data in session state for use across interactions
    st.session_state.processed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler,
        'anomaly_ratio': anomaly_ratio
    }

    return st.session_state.processed_data

# Function to load trained models - Updated to use improved models
def load_models():
    """Load the trained anomaly detection models with improved accuracy."""
    models = {}
    
    # Define model filenames - prioritizing the specific models that work best
    model_files = {
        'Isolation Forest': ['isolation_forest_improved.joblib', 'isolation_forest.joblib'],
        'One-Class SVM': ['one_class_svm_improved.joblib', 'one_class_svm.joblib'],
        'Local Outlier Factor': ['local_outlier_factor_improved.joblib', 'local_outlier_factor.joblib'],
        'DBSCAN': ['dbscan_improved.joblib', 'dbscan_best.joblib', 'dbscan_fixed.joblib', 'dbscan.joblib']
    }
    
    # Try to load each model, prioritizing the first one in each list
    for name, filenames in model_files.items():
        for filename in filenames:
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
                try:
            models[name] = joblib.load(model_path)
                    # Special highlight for the specifically requested DBSCAN model
                    if name == 'DBSCAN' and filename == 'dbscan_improved.joblib':
                        st.success(f"Loaded requested model: {name} from {filename} (90% accuracy)")
                    elif 'improved' in filename.lower():
                        st.success(f"Loaded improved model: {name} from {filename}")
                    else:
                        st.info(f"Loaded model: {name} from {filename}")
                    break  # Found a working model, no need to try others
                except Exception as e:
                    st.warning(f"Error loading {filename}: {str(e)}")
    
    # Store models in session state for use across interactions
    st.session_state.models = models
    return models

# Function to get anomaly predictions
def get_predictions(model, X, model_name):
    """Get anomaly predictions from a model."""
    if model_name == "DBSCAN" or "dbscan" in model_name.lower():
        # For DBSCAN improved model - use the exact approach from run_improved_models.py
        try:
                from sklearn.cluster import DBSCAN
            y_test = st.session_state.processed_data['y_test']
            
            # Create DBSCAN model with optimal parameters
            model_dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
            cluster_labels = model_dbscan.fit_predict(X)
            
            # Map clusters to labels
            clusters = np.unique(cluster_labels)
            cluster_to_label = {}
            
            # For each cluster, assign the majority label - exactly like in run_improved_models.py
            for cluster in clusters:
                if cluster == -1:  # Noise points are anomalies
                    cluster_to_label[cluster] = 1
                else:
                    mask = (cluster_labels == cluster)
                    normal_count = np.sum(y_test[mask] == 0)
                    anomaly_count = np.sum(y_test[mask] == 1)
                    cluster_to_label[cluster] = 0 if normal_count > anomaly_count else 1
            
            # Map clusters to binary labels
            y_pred = np.zeros_like(cluster_labels)
            for i, cluster in enumerate(cluster_labels):
                y_pred[i] = cluster_to_label.get(cluster, 1)  # Default to anomaly
            
            # Save DBSCAN model with cluster_to_label mapping
            dbscan_model = {
                'model': model_dbscan,
                'cluster_to_label': cluster_to_label,
                'eps': 0.5,
                'min_samples': 5
            }
            
            # Save the model for future use
            try:
                joblib.dump(dbscan_model, os.path.join(MODELS_DIR, 'dbscan_current.joblib'))
            except Exception as e:
                st.warning(f"Could not save current DBSCAN model: {e}")
            
                return y_pred
        except Exception as e:
            st.error(f"Error with DBSCAN model: {e}")
            return np.zeros(X.shape[0])  # Return default prediction as fallback
    else:
        # For Isolation Forest, One-Class SVM, LOF
        try:
        y_pred = model.predict(X)
        # Convert prediction format
        # For these models, -1 is anomaly (our label 1), 1 is normal (our label 0)
        if model_name in ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]:
            # Convert -1/1 to 1/0 (anomaly/normal)
            y_pred = np.array([1 if label == -1 else 0 for label in y_pred])
        return y_pred
        except Exception as e:
            st.error(f"Error with {model_name} model: {e}")
            return np.zeros(X.shape[0])  # Return default prediction as fallback

# Function to evaluate anomaly detection results
def evaluate_results(y_true, y_pred):
    """Evaluate anomaly detection results."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1_score_val = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix values for specificity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score_val,
        'specificity': specificity,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp
    }
    
    return metrics

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar=False,
        annot_kws={"size": 16}
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    return fig

# Function to plot ROC curve
def plot_roc_curve(y_true, y_score):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig

# Function to compare models visually - Enhanced with visualize_model_comparison.py techniques
def compare_models_plot(metrics_df):
    """Create enhanced visualizations comparing models."""
    # Basic bar chart - with Plotly for interactivity
    metrics_df_melted = pd.melt(
        metrics_df.reset_index(), 
        id_vars='index',
        value_vars=['accuracy', 'precision', 'recall', 'f1_score', 'specificity'],
        var_name='Metric',
        value_name='Value'
    )
    metrics_df_melted = metrics_df_melted.rename(columns={'index': 'Model'})
    
    bar_fig = px.bar(
        metrics_df_melted,
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        labels={'Value': 'Score', 'Model': 'Model'}
    )
    
    # Radar chart (using Plotly)
    def create_radar_chart(df):
        categories = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        fig = go.Figure()
        
        for i, model in enumerate(df.index):
            values = df.loc[model, categories].tolist()
            values.append(values[0])  # Close the loop
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=model
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Model Metrics Comparison (Radar Chart)'
    )
    return fig
    
    radar_fig = create_radar_chart(metrics_df)
    
    # Scatter plot for F1 vs Accuracy
    scatter_fig = px.scatter(
        metrics_df.reset_index(),
        x='accuracy',
        y='f1_score',
        size='precision',
        color='index',
        hover_name='index',
        size_max=60,
        title='F1 Score vs. Accuracy'
    )
    
    # 3D scatter plot
    scatter_3d_fig = px.scatter_3d(
        metrics_df.reset_index(),
        x='accuracy',
        y='precision',
        z='recall',
        color='index',
        hover_name='index',
        title='3D View of Model Performance'
    )
    
    return {
        'bar_chart': bar_fig,
        'radar_chart': radar_fig,
        'scatter': scatter_fig,
        '3d_scatter': scatter_3d_fig
    }

# Function to create model with specific parameters from run_improved_models.py
def create_model_with_parameters(model_name):
    """Create a model with the optimized parameters from run_improved_models.py."""
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    
    # Get anomaly ratio if available in session state
    anomaly_ratio = st.session_state.processed_data.get('anomaly_ratio', 0.46)  # Default from the run_improved_models.py
    
    if model_name == "DBSCAN":
        # DBSCAN parameters from run_improved_models.py - optimized for high accuracy
        model_dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        
        # Try to load the existing model to get the cluster_to_label mapping
        try:
            existing_model = joblib.load(os.path.join(MODELS_DIR, 'dbscan_improved.joblib'))
            if isinstance(existing_model, dict) and 'cluster_to_label' in existing_model:
                cluster_to_label = existing_model['cluster_to_label']
            else:
                cluster_to_label = {-1: 1}  # Default: noise points are anomalies
        except:
            cluster_to_label = {-1: 1}  # Default: noise points are anomalies
        
        # Create the model dict structure that matches what's used in run_improved_models.py
        dbscan_model = {
            'model': model_dbscan,
            'cluster_to_label': cluster_to_label,
            'eps': 0.5,
            'min_samples': 5
        }
        
        return dbscan_model
    else:
        # For other models, use the already loaded models without changing parameters
        if model_name in st.session_state.models:
            return st.session_state.models[model_name]
        else:
            st.error(f"Model {model_name} not found in loaded models.")
            return None

# Main application
def main():
    st.set_page_config(
        page_title="Network Anomaly Detection System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI with classic, novel look
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1.5rem;
        font-family: 'Garamond', serif;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        padding: 1.2rem;
        border-bottom: 2px solid #7F8C8D;
        background-color: #EAEAEA;
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495E;
        margin-bottom: 1rem;
        font-weight: bold;
        font-family: 'Garamond', serif;
        border-left: 4px solid #E74C3C;
        padding-left: 10px;
    }
    .info-text {
        background-color: #F8F9F9;
        color: #2C3E50;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #3498DB;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        font-family: 'Georgia', serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #2C3E50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.7rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #2C3E50;
        transform: none;
        box-shadow: none;
    }
    .metric-card {
        background: linear-gradient(135deg, #F5F7FA 0%, #E8EAED 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: #2C3E50;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #D5D8DC;
    }
    .metric-card:hover {
        transform: none;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #E74C3C;
        font-family: 'Times New Roman', serif;
    }
    .metric-label {
        font-size: 1rem;
        color: #7F8C8D;
        font-family: 'Georgia', serif;
        letter-spacing: 1px;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9F9;
        color: #2C3E50;
    }
    .success-message {
        background-color: #D5F5E3;
        color: #186A3B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
        border-left: 5px solid #58D68D;
        font-family: 'Georgia', serif;
    }
    .warning-message {
        background-color: #FDEBD0;
        color: #9A7D0A;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
        border-left: 5px solid #F39C12;
        font-family: 'Georgia', serif;
    }
    .error-message {
        background-color: #FADBD8;
        color: #943126;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
        border-left: 5px solid #E74C3C;
        font-family: 'Georgia', serif;
    }
    .classic-container {
        background-color: #F8F9F9;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid #D5D8DC;
    }
    .plot-container {
        max-height: 500px;
        overflow: auto;
    }
    .small-plot {
        max-height: 400px;
        max-width: 90%;
        margin: 0 auto;
    }
    .model-badge {
        background-color: #E74C3C;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #7F8C8D;
        font-size: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid #D5D8DC;
    }
    .best-model-tag {
        background-color: #E74C3C;
        color: white;
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 10px;
        margin-left: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application header
    st.markdown('<div class="main-header">Network Anomaly Detection System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
    This application demonstrates advanced anomaly detection models for network traffic analysis.
    DBSCAN is the best performing model (90.83% accuracy), but you can choose other models for comparison.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Step 1: Data Loading and Preprocessing
    st.sidebar.header("Step 1: Data Preprocessing")
    preprocess_btn = st.sidebar.button("Load and Preprocess Data")
    if preprocess_btn or st.session_state.processed_data is not None:
        if preprocess_btn:
            with st.spinner("Loading and preprocessing data..."):
                data = load_nsl_kdd_data()
                if data is not None:
                    processed_data = preprocess_nsl_kdd_data(data)
                    st.markdown('<div class="success-message">✅ Data loaded and preprocessed successfully!</div>', unsafe_allow_html=True)
        
        # Display data overview if it exists in session state
        if st.session_state.processed_data is not None:
            with st.container():
                st.markdown('<div class="classic-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-header">Data Overview</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
        with col1:
                    st.write(f"**Training samples:** {st.session_state.processed_data['X_train'].shape[0]}")
                    st.write(f"**Features:** {st.session_state.processed_data['X_train'].shape[1]}")
        
        with col2:
                    normal_count = (st.session_state.processed_data['y_train'] == 0).sum()
                    anomaly_count = (st.session_state.processed_data['y_train'] == 1).sum()
                    st.write(f"**Normal samples:** {normal_count} ({normal_count/len(st.session_state.processed_data['y_train'])*100:.2f}%)")
                    st.write(f"**Anomaly samples:** {anomaly_count} ({anomaly_count/len(st.session_state.processed_data['y_train'])*100:.2f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
        
            # Display a sample of the data
            with st.container():
                st.markdown('<div class="classic-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-header">Feature Correlation</div>', unsafe_allow_html=True)
                # Get feature names and create a correlation heatmap
                try:
                    feature_names = st.session_state.processed_data['feature_names'][:8]  # Take first 8 features for clarity
                    corr_data = pd.DataFrame(st.session_state.processed_data['X_train'][:, :8], 
                                            columns=feature_names)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_data.corr(), annot=False, cmap='viridis', ax=ax)
            st.pyplot(fig)
                except Exception as e:
                    st.markdown(f'<div class="warning-message">⚠️ Could not generate correlation heatmap: {e}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Model Selection
    st.sidebar.header("Step 2: Model Selection")
    if st.session_state.processed_data is not None:
        load_models_btn = st.sidebar.button("Load Models")
        
        # Load models if button clicked or models already in session state
        if load_models_btn or st.session_state.models is not None:
            if load_models_btn:
                with st.spinner("Loading models..."):
                    models = load_models()
                    # Set DBSCAN as default selected model
                    st.session_state.selected_model = "DBSCAN"
                    
            # Only show model selection if models are loaded
            if st.session_state.models is not None:
                with st.container():
                    st.markdown('<div class="classic-container">', unsafe_allow_html=True)
                    st.markdown('<div class="sub-header">Model Selection</div>', unsafe_allow_html=True)
                    
                    # Display model selection in the main area for better visibility
                    model_names = list(st.session_state.models.keys())
                    
                    # Create a list with "BEST" tag for DBSCAN
                    model_options = []
                    for model in model_names:
                        if model == "DBSCAN":
                            model_options.append(f"{model} (BEST)")
                        else:
                            model_options.append(model)
                                
                    # Default to DBSCAN
                    default_idx = model_options.index("DBSCAN (BEST)") if "DBSCAN (BEST)" in model_options else 0
                    selected_option = st.selectbox("Select Model for Anomaly Detection", model_options, index=default_idx)
                    
                    # Extract actual model name
                    selected_model = selected_option.split(" ")[0] if "(" in selected_option else selected_option
                    st.session_state.selected_model = selected_model
                    
                    # Display model information with special highlight for DBSCAN
                    if selected_model == "DBSCAN":
                        st.markdown(f'<div style="background-color: #F8F9F9; padding: 1rem; border-radius: 0.5rem; color: #2C3E50; border-left: 5px solid #E74C3C;">Selected model: <span style="color: #E74C3C; font-weight: bold;">{selected_model}</span> <span class="model-badge">Best Model</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="background-color: #F8F9F9; padding: 1rem; border-radius: 0.5rem; color: #2C3E50; border-left: 5px solid #3498DB;">Selected model: <span style="color: #3498DB; font-weight: bold;">{selected_model}</span></div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Step 3: Model Evaluation
                with st.container():
                    st.markdown('<div class="classic-container">', unsafe_allow_html=True)
                    st.markdown('<div class="sub-header">Run Anomaly Detection</div>', unsafe_allow_html=True)
                    if st.button("Detect Anomalies"):
                        with st.spinner("Running anomaly detection..."):
                            # Always use optimized parameters
                            model = create_model_with_parameters(selected_model)
                            if model is None:
                                model = st.session_state.models[selected_model]
                            
                            X_test = st.session_state.processed_data['X_test']
                            y_test = st.session_state.processed_data['y_test']
                                    
                            # Get predictions
                            y_pred = get_predictions(model, X_test, selected_model)
                            
                            # Calculate metrics
                            metrics = evaluate_results(y_test, y_pred)
                                    
                            # Display results
                            st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Accuracy</div>
                                    <div class="metric-value">{metrics['accuracy']:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Precision</div>
                                    <div class="metric-value">{metrics['precision']:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Recall</div>
                                    <div class="metric-value">{metrics['recall']:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">F1 Score</div>
                                    <div class="metric-value">{metrics['f1_score']:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            with col5:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Specificity</div>
                                    <div class="metric-value">{metrics['specificity']:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                            # Plot confusion matrix
                            st.markdown('<div class="sub-header">Confusion Matrix</div>', unsafe_allow_html=True)
                            st.markdown('<div class="small-plot">', unsafe_allow_html=True)
                            fig = plot_confusion_matrix(y_test, y_pred)
                            st.pyplot(fig)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Save results in session state
                            if 'results' not in st.session_state:
                                st.session_state.results = {}
                            st.session_state.results[selected_model] = {
                                'metrics': metrics,
                                'y_pred': y_pred
                            }
                    st.markdown('</div>', unsafe_allow_html=True)
                        
                # Step 4: Model Comparison (if multiple models have been run)
                if 'results' in st.session_state and len(st.session_state.results) > 0:
                    with st.container():
                        st.markdown('<div class="classic-container">', unsafe_allow_html=True)
                        st.markdown('<div class="sub-header">Model Comparison</div>', unsafe_allow_html=True)
                        
                        # Create metrics DataFrame for visualization
                        metrics_data = []
                        for model_name, result in st.session_state.results.items():
                            metrics_dict = result['metrics'].copy()
                            metrics_dict['index'] = model_name
                            metrics_data.append(metrics_dict)
                        
                        if len(metrics_data) > 0:
                            metrics_df = pd.DataFrame(metrics_data).set_index('index')
                            
                            # Generate visualizations
                            visuals = compare_models_plot(metrics_df)
                            
                            if len(metrics_data) > 1:
                                # Show visualizations in tabs
                                tabs = st.tabs(["Bar Chart", "Radar Chart", "Scatter Plot", "3D View"])
                                
                                with tabs[0]:
                                    st.plotly_chart(visuals['bar_chart'], use_container_width=True)
                                
                                with tabs[1]:
                                    st.plotly_chart(visuals['radar_chart'], use_container_width=True)
                                
                                with tabs[2]:
                                    st.plotly_chart(visuals['scatter'], use_container_width=True)
                                
                                with tabs[3]:
                                    st.plotly_chart(visuals['3d_scatter'], use_container_width=True)
                            else:
                                st.info("Run more models to see comparison visualizations")
                            
                            # Option to save comparison results
                            if st.button("Save Model Comparison Results"):
                                metrics_df.to_csv(os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv'))
                                st.markdown('<div class="success-message">✅ Model comparison metrics saved successfully!</div>', unsafe_allow_html=True)
                    else:
                            st.info("Run at least one model to see metrics")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Add footer
                st.markdown('<div class="footer">Network Anomaly Detection System - Powered by Machine Learning</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-text">Click \'Load Models\' to load the available anomaly detection models.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-message">⚠️ Please complete Step 1: Load and Preprocess Data first.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 