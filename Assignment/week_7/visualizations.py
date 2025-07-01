import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import BreastCancerModel

def plot_model_performance(model, X_test, y_test):
    """
    Create visualizations for model performance metrics
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of plotly figures
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figures dictionary
    figures = {}
    
    # 1. Confusion Matrix
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=['Malignant (0)', 'Benign (1)'],
        y=['Malignant (0)', 'Benign (1)'],
        color_continuous_scale='Blues',
        title="Confusion Matrix"
    )
    fig_cm.update_layout(
        width=500,
        height=500
    )
    figures['confusion_matrix'] = fig_cm
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig_roc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig_roc.update_layout(
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.05]
    )
    figures['roc_curve'] = fig_roc
    
    return figures

def plot_feature_distributions(dataset=None):
    """
    Create visualizations for feature distributions
    
    Parameters:
        dataset: Optional breast cancer dataset
        
    Returns:
        Dictionary of plotly figures
    """
    if dataset is None:
        dataset = load_breast_cancer()
    
    # Create DataFrame from dataset
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    
    # Create figures dictionary
    figures = {}
    
    # 1. Class Distribution
    class_counts = df['diagnosis'].value_counts().reset_index()
    class_counts.columns = ['Diagnosis', 'Count']
    
    fig_class = px.pie(
        class_counts,
        values='Count',
        names='Diagnosis',
        title='Distribution of Benign vs Malignant Cases',
        color='Diagnosis',
        color_discrete_map={'Benign': 'green', 'Malignant': 'red'}
    )
    fig_class.update_traces(textposition='inside', textinfo='percent+label')
    fig_class.update_layout(
        width=500,
        height=500
    )
    figures['class_distribution'] = fig_class
    
    # 2. Feature Correlation Heatmap
    corr = df.iloc[:, :-2].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.mask(mask)
    
    fig_corr = px.imshow(
        corr_masked,
        color_continuous_scale='viridis',
        title='Feature Correlation Heatmap'
    )
    fig_corr.update_layout(
        width=800,
        height=800
    )
    figures['correlation'] = fig_corr
    
    # 3. Top Features Comparison by Diagnosis
    # Select a few top features for comparison
    top_features = ['mean radius', 'mean texture', 'mean perimeter', 
                   'mean area', 'mean smoothness', 'mean compactness']
    
    fig_features = make_subplots(
        rows=2, cols=3,
        subplot_titles=top_features
    )
    
    for i, feature in enumerate(top_features):
        row = i // 3 + 1
        col = i % 3 + 1
        
        # Add boxplot for each feature
        for diagnosis, color in zip(['Malignant', 'Benign'], ['red', 'green']):
            feature_data = df[df['diagnosis'] == diagnosis][feature]
            
            fig_features.add_trace(
                go.Box(
                    y=feature_data,
                    name=diagnosis,
                    marker_color=color,
                    showlegend=True if i == 0 else False
                ),
                row=row, col=col
            )
    
    fig_features.update_layout(
        height=600,
        width=900,
        title_text="Key Features by Diagnosis",
        boxmode='group'
    )
    figures['feature_comparison'] = fig_features
    
    return figures

def create_prediction_input_ui(feature_info):
    """
    Create UI elements for feature input
    
    Parameters:
        feature_info: Dictionary with feature range information
        
    Returns:
        Dictionary of input values
    """
    st.write("### Patient Data Input")
    
    # Create a dictionary to store input values
    input_values = {}
    
    # Select important features to display (to avoid overwhelming the user)
    important_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity', 
        'mean concave points', 'mean symmetry', 'mean fractal dimension'
    ]
    
    # Create sliders for each feature
    for feature in important_features:
        feature_range = feature_info[feature]
        min_val = feature_range['min']
        max_val = feature_range['max']
        mean_val = feature_range['mean']
        
        input_values[feature] = st.slider(
            feature, 
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(mean_val),
            format="%.6f"
        )
    
    # For remaining features, use default (mean) values
    for feature in feature_info:
        if feature not in important_features:
            input_values[feature] = feature_info[feature]['mean']
    
    return input_values 

def create_visualization_folder():
    """Create visualizations folder if it doesn't exist"""
    os.makedirs('visualizations', exist_ok=True)

def save_feature_importance(model_instance, save_path='visualizations/feature_importance.png'):
    """Generate and save feature importance visualization"""
    # Get feature importance for random forest
    model_instance.set_model("Random Forest")
    importance_df = model_instance.get_feature_importance(top_n=10)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values('Importance'), palette='viridis')
    plt.title('Top 10 Feature Importance (Random Forest)', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance visualization saved to {save_path}")

def save_model_comparison(model_instance, save_path='visualizations/model_comparison.png'):
    """Generate and save model comparison visualization"""
    # Get metrics for all models
    metrics_df = model_instance.get_all_models_metrics()
    
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(
        metrics_df, 
        id_vars=['model'], 
        value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create figure
    plt.figure(figsize=(14, 10))
    chart = sns.barplot(x='model', y='Value', hue='Metric', data=melted_df, palette='Set1')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison visualization saved to {save_path}")

def save_roc_curves(model_instance, save_path='visualizations/roc_curves.png'):
    """Generate and save ROC curves for all models"""
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot ROC curve for each model
    for name, model in model_instance.models.items():
        # Get predictions
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Add reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves Comparison', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves visualization saved to {save_path}")

def save_confusion_matrices(model_instance, save_path='visualizations/confusion_matrices.png'):
    """Generate and save confusion matrices for all models"""
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot confusion matrix for each model
    for i, (name, model) in enumerate(model_instance.models.items()):
        # Get predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            ax=axes[i]
        )
        axes[i].set_title(f'{name}', fontsize=14)
        axes[i].set_xlabel('Predicted Label', fontsize=12)
        axes[i].set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices visualization saved to {save_path}")

def save_prediction_gauge(save_path='visualizations/prediction_gauge.png'):
    """Generate and save a sample prediction gauge chart using Matplotlib"""
    # Create a sample gauge for both outcomes
    for prediction, prob, label in [(0, 0.85, 'malignant'), (1, 0.92, 'benign')]:
        # Define colors
        color = "#b91c1c" if label == 'malignant' else "#86efac"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
        
        # Set gauge properties
        theta = np.linspace(0, 180, 100) * np.pi / 180  # Half circle
        r = np.ones_like(theta)
        
        # Background
        ax.fill_between(theta, 0, r, color='lightgray', alpha=0.3)
        
        # Value indicator
        value_theta = prob * np.pi
        value_idx = int(prob * 100)
        ax.fill_between(theta[:value_idx], 0, r[:value_idx], color=color, alpha=0.8)
        
        # Add value text
        fig.text(0.5, 0.25, f"{prob*100:.1f}%", fontsize=24, ha='center')
        fig.text(0.5, 0.15, f"Prediction: {label.capitalize()}", fontsize=16, ha='center')
        
        # Customize chart
        ax.set_rticks([])  # No radial ticks
        
        # Fixed theta grids - ensure number of labels matches number of ticks
        thetas = np.arange(0, 181, 30) * np.pi / 180
        labels = ['0%', '30%', '60%', '90%', '120%', '150%', '180%']
        ax.set_xticks(thetas)
        ax.set_xticklabels(labels)
        
        ax.grid(True, alpha=0.3)
        ax.spines['polar'].set_visible(False)
        
        # Save to file
        plt.tight_layout()
        plt.savefig(f'visualizations/prediction_gauge_{label}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction gauge for {label} saved to visualizations/prediction_gauge_{label}.png")

def save_class_distribution(save_path='visualizations/class_distribution.png'):
    """Generate and save the class distribution visualization"""
    # Load dataset
    data = load_breast_cancer()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=data.target, palette=['#b91c1c', '#86efac'])
    
    # Add labels and counts
    ax.set_xticklabels(['Malignant', 'Benign'])
    plt.title('Distribution of Breast Cancer Classes', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class distribution visualization saved to {save_path}")

def save_correlation_heatmap(save_path='visualizations/correlation_heatmap.png'):
    """Generate and save correlation heatmap of features"""
    # Load dataset
    data = load_breast_cancer()
    
    # Convert to DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Calculate correlation
    corr = df.corr()
    
    # Create figure
    plt.figure(figsize=(16, 14))
    mask = np.triu(corr)
    sns.heatmap(
        corr, 
        mask=mask,
        cmap='coolwarm', 
        annot=False, 
        square=True, 
        linewidths=.5,
        vmin=-1, 
        vmax=1, 
        center=0,
        cbar_kws={"shrink": .8}
    )
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to {save_path}")

def generate_all_visualizations():
    """Generate all visualizations for the project"""
    # Create visualization folder
    create_visualization_folder()
    
    # Initialize model
    model_instance = BreastCancerModel()
    
    # Train all models if not already trained
    if len(model_instance.models) <= 1:
        model_instance.train_additional_models()
    
    # Generate visualizations
    save_feature_importance(model_instance)
    save_model_comparison(model_instance)
    save_roc_curves(model_instance)
    save_confusion_matrices(model_instance)
    save_prediction_gauge()
    save_class_distribution()
    save_correlation_heatmap()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    generate_all_visualizations() 