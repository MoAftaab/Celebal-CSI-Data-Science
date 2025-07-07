import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import sys
from sklearn.datasets import load_breast_cancer

# Import custom modules
from utils import BreastCancerModel, plot_feature_importance, plot_prediction_gauge, get_feature_explanation, plot_model_comparison
from visualizations import plot_model_performance, plot_feature_distributions, create_prediction_input_ui

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4b9fff;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .description {
        font-size: 1.1rem;
        text-align: justify;
        margin-bottom: 2rem;
        color: #e0e0e0;
        color: #e0e0e0;
    }
    
    .highlight {
        background-color: #222222;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #333333;
    }
    
    .prediction-benign {
        color: #00401A;
        font-weight: bold;
        font-size: 1.5rem;
        background-color: #86efac;
        padding: 0.5em 0.7em;
        border-radius: 0.3em;
        display: inline-block;
        margin-bottom: 0.5em;
    }
    
    .prediction-malignant {
        color: #FFFFFF;
        font-weight: bold;
        font-size: 1.5rem;
        background-color: #b91c1c;
        padding: 0.5em 0.7em;
        border-radius: 0.3em;
        display: inline-block;
        margin-bottom: 0.5em;
    }
    
    .confidence-box {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        padding: 0.5em 1em;
        border-radius: 0.3em;
        display: inline-block;
        margin-top: 0.2em;
    }
    
    .accuracy-metric {
        font-size: 1.2rem;
        font-weight: bold;
        color: #4b9fff;
    }
    
    .model-selector {
        padding: 1rem;
        background-color: #111111;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #333333;
    }
    
    .model-comparison {
        padding: 1rem;
        background-color: #111111;
        border-radius: 0.5rem;
        margin-top: 2rem;
        border: 1px solid #333333;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_model():
    """Load and cache the model to avoid reloading on each run"""
    return BreastCancerModel()

@st.cache_data
def load_test_data():
    """Load and cache test data for visualizations"""
    # Load breast cancer dataset directly
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    return X_test, y_test

def main():
    # Header
    st.markdown("<h1 class='main-header'>Breast Cancer Classifier</h1>", unsafe_allow_html=True)
    
    # Description
    st.markdown(
        """
        <p class='description'>
        This web application uses machine learning models to predict whether a breast mass is benign or malignant
        based on features extracted from a digitized image of a fine needle aspirate (FNA) of a breast mass.
        The model was trained on the Wisconsin Breast Cancer Dataset and achieves high accuracy in classifying tumors.
        </p>
        """, 
        unsafe_allow_html=True
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # Sidebar
    st.sidebar.image("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.png", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Model Performance", "Data Exploration", "Model Comparison"])
    
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app demonstrates the deployment of a machine learning model using Streamlit.
        
        **Dataset:** Wisconsin Breast Cancer Dataset
        
        **Models:** 
        - Random Forest
        - Gradient Boosting
        - Logistic Regression
        - SVM
        - K-Nearest Neighbors
        - Naive Bayes
        
        **Task:** Binary Classification (Malignant vs. Benign)
        """
    )
    
    # Content based on selected page
    if page == "Prediction":
        prediction_page(model)
    elif page == "Model Performance":
        performance_page(model)
    elif page == "Data Exploration":
        exploration_page(model)
    else:
        comparison_page(model)

def prediction_page(model):
    st.markdown("<h2 class='sub-header'>Cancer Prediction</h2>", unsafe_allow_html=True)
    
    # Model selection
    st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
    st.markdown("### Select Model")
    st.markdown("Choose a machine learning model for prediction:")
    
    # List available models
    model_options = ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "K-Nearest Neighbors", "Naive Bayes"]
    selected_model = st.selectbox("Model", model_options)
    
    # Update the active model
    if selected_model != model.model_name:
        with st.spinner(f"Loading {selected_model} model..."):
            model.set_model(selected_model)
        st.success(f"{selected_model} model loaded successfully!")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Use sample data", "Enter values manually"]
    )
    
    if input_method == "Use sample data":
        # Get sample data
        samples = model.get_sample_data(n_samples=5)
        
        # Remove target column from display
        samples_display = samples.drop('target', axis=1)
        
        # Select a sample
        selected_idx = st.selectbox(
            "Select a sample from the dataset:",
            range(len(samples)),
            format_func=lambda i: f"Sample {i+1} (Actual: {'Benign' if samples.iloc[i]['target'] == 1 else 'Malignant'})"
        )
        
        # Display selected sample
        with st.expander("View sample data", expanded=False):
            st.dataframe(samples_display.iloc[[selected_idx]])
        
        # Get input data from selected sample
        input_data = samples_display.iloc[[selected_idx]]
        
        # Display actual diagnosis
        actual = "Benign" if samples.iloc[selected_idx]['target'] == 1 else "Malignant"
        st.markdown(f"<div class='highlight'>Actual diagnosis: {'<span class=\"prediction-benign\">Benign</span>' if actual == 'Benign' else '<span class=\"prediction-malignant\">Malignant</span>'}</div>", unsafe_allow_html=True)
        
    else:
        # Manual input through sliders
        feature_info = model.get_feature_ranges()
        input_values = create_prediction_input_ui(feature_info)
        
        # Convert to DataFrame
        input_data = pd.DataFrame([input_values])
    
    # Make prediction
    if st.button("Predict", type="primary"):
        with st.spinner("Making prediction..."):
            prediction, probability = model.predict(input_data)
            
            # Display prediction
            prediction_label = "Benign" if prediction == 1 else "Malignant"
            prediction_class = "prediction-benign" if prediction == 1 else "prediction-malignant"
            
            st.markdown(
                f"""
                <div class='highlight'>
                    <h3>Prediction Result</h3>
                    <div class='{prediction_class}'>The tumor is predicted to be: {prediction_label}</div>
                    <div class='confidence-box'>with {probability:.2%} confidence</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display gauge chart
            st.plotly_chart(plot_prediction_gauge(probability, prediction))
            
            # Display feature importance and explanation
            st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
            importance_df = model.get_feature_importance()
            
            st.plotly_chart(plot_feature_importance(importance_df))
            
            if importance_df is not None:
                st.markdown("<h4>Top Features Explanation</h4>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                for idx, row in importance_df.head(5).iterrows():
                    feature = row['Feature']
                    if feature in input_data.columns:
                        value = input_data[feature].values[0]
                        explanation = get_feature_explanation(feature, value, importance_df)
                        st.markdown(f"**{feature}**: {value:.6f}<br>{explanation}", unsafe_allow_html=True)
                        st.divider()

def performance_page(model):
    st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
    
    # Model selection
    st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
    st.markdown("### Select Model for Evaluation")
    
    # List available models
    model_options = ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "K-Nearest Neighbors", "Naive Bayes"]
    selected_model = st.selectbox("Model", model_options)
    
    # Update the active model
    if selected_model != model.model_name:
        with st.spinner(f"Loading {selected_model} model..."):
            model.set_model(selected_model)
        st.success(f"{selected_model} model loaded successfully!")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Load the dataset
    dataset = load_breast_cancer()
    
    # Create model metrics section
    st.markdown("### Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    # Get metrics for the current model
    metrics_df = model.get_all_models_metrics()
    current_metrics = metrics_df[metrics_df['model'] == selected_model].iloc[0]
    
    with col1:
        st.markdown(
            f"""
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; text-align:center;">
                <h4>Accuracy</h4>
                <p class="accuracy-metric">{current_metrics['accuracy']:.1%}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; text-align:center;">
                <h4>Precision</h4>
                <p class="accuracy-metric">{current_metrics['precision']:.1%}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; text-align:center;">
                <h4>Recall</h4>
                <p class="accuracy-metric">{current_metrics['recall']:.1%}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; text-align:center;">
                <h4>F1 Score</h4>
                <p class="accuracy-metric">{current_metrics['f1']:.1%}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Show confusion matrix and ROC curve
    try:
        # Try to load test data
        X_test, y_test = load_test_data()
        
        # Plot model performance
        figures = plot_model_performance(model.model, X_test, y_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(figures['confusion_matrix'])
        
        with col2:
            st.plotly_chart(figures['roc_curve'])
    
    except Exception as e:
        st.warning(f"Could not load test data for performance visualization: {e}")
        st.info("Displaying metrics from internal evaluation.")
    
    # Model details
    with st.expander("Model Architecture Details"):
        if selected_model == "Random Forest":
            st.markdown(
                """
                ### Random Forest Classifier
                
                The Random Forest model uses an ensemble of decision trees with the following hyperparameters:
                
                - **n_estimators**: 100 (Number of trees in the forest)
                - **max_depth**: 10 (Maximum depth of each tree)
                - **random_state**: 42 (For reproducibility)
                
                Random Forests work by creating multiple decision trees on different subsamples of the data and features, 
                then averaging their predictions for improved accuracy and reduced overfitting.
                """
            )
        elif selected_model == "Gradient Boosting":
            st.markdown(
                """
                ### Gradient Boosting Classifier
                
                The Gradient Boosting model uses a sequential ensemble of decision trees with the following hyperparameters:
                
                - **n_estimators**: 100 (Number of boosting stages)
                - **max_depth**: 4 (Maximum depth of each tree)
                - **random_state**: 42 (For reproducibility)
                
                Gradient Boosting builds trees sequentially, with each tree correcting the errors of the previous ones,
                resulting in a powerful model that often achieves high accuracy.
                """
            )
        elif selected_model == "Logistic Regression":
            st.markdown(
                """
                ### Logistic Regression
                
                The Logistic Regression model uses the following hyperparameters:
                
                - **max_iter**: 1000 (Maximum number of iterations for convergence)
                - **random_state**: 42 (For reproducibility)
                
                Logistic Regression is a linear model that estimates the probability of the binary outcome
                using a logistic function. It's simple, interpretable, and works well for linearly separable data.
                """
            )
        elif selected_model == "SVM":
            st.markdown(
                """
                ### Support Vector Machine (SVM)
                
                The SVM model uses the following hyperparameters:
                
                - **kernel**: rbf (Radial Basis Function kernel)
                - **probability**: True (Enables probability estimates)
                - **random_state**: 42 (For reproducibility)
                
                SVM works by finding the hyperplane that best separates the classes in a high-dimensional space.
                The RBF kernel allows it to capture non-linear relationships in the data.
                """
            )
        elif selected_model == "K-Nearest Neighbors":
            st.markdown(
                """
                ### K-Nearest Neighbors (KNN)
                
                The KNN model uses the following hyperparameters:
                
                - **n_neighbors**: 5 (Number of neighbors to consider)
                
                KNN makes predictions based on the majority class of the k nearest data points.
                It's a non-parametric method that can capture complex decision boundaries.
                """
            )
        else:  # Naive Bayes
            st.markdown(
                """
                ### Gaussian Naive Bayes
                
                The Naive Bayes model assumes features follow a Gaussian distribution and are conditionally independent.
                
                Naive Bayes is based on Bayes' theorem and works well with high-dimensional data.
                It's particularly useful when the independence assumption approximately holds.
                """
            )

def exploration_page(model):
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    # Create visualizations
    figures = plot_feature_distributions()
    
    # Class distribution
    st.markdown("### Dataset Class Distribution")
    st.plotly_chart(figures['class_distribution'])
    
    # Feature comparison
    st.markdown("### Feature Comparison by Diagnosis")
    st.plotly_chart(figures['feature_comparison'])
    
    # Correlation heatmap
    st.markdown("### Feature Correlation")
    st.plotly_chart(figures['correlation'])
    
    # Dataset description
    with st.expander("Dataset Description"):
        st.markdown(
            """
            ### Wisconsin Breast Cancer Dataset
            
            The Wisconsin Breast Cancer dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.
            
            **Number of Instances**: 569
            
            **Number of Features**: 30
            
            **Target Classes**:
            - Malignant (0)
            - Benign (1)
            
            **Features**:
            - **Radius**: mean of distances from center to points on the perimeter
            - **Texture**: standard deviation of gray-scale values
            - **Perimeter**: perimeter of the tumor
            - **Area**: area of the tumor
            - **Smoothness**: local variation in radius lengths
            - **Compactness**: perimeter^2 / area - 1.0
            - **Concavity**: severity of concave portions of the contour
            - **Concave points**: number of concave portions of the contour
            - **Symmetry**: symmetry of the tumor
            - **Fractal dimension**: "coastline approximation" - 1
            
            For each feature, the mean, standard error, and "worst" (mean of the three largest values) are computed, resulting in 30 features.
            """
        )

def comparison_page(model):
    st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <p class='description'>
        This page compares the performance of different machine learning models on the breast cancer classification task.
        You can see how different algorithms perform in terms of accuracy, precision, recall, F1 score, and ROC AUC.
        </p>
        """, 
        unsafe_allow_html=True
    )
    
    # Ensure all models are trained
    with st.spinner("Training and evaluating all models..."):
        metrics_df = model.get_all_models_metrics()
    
    # Display comparison chart
    st.plotly_chart(plot_model_comparison(metrics_df))
    
    # Display metrics table
    st.markdown("### Detailed Performance Metrics")
    
    # Format metrics for display
    display_df = metrics_df.copy()
    for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
    
    # Rename columns for better display
    display_df = display_df.rename(columns={
        'model': 'Model',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'roc_auc': 'ROC AUC'
    })
    
    # Sort by F1 score (descending)
    display_df = display_df.sort_values('F1 Score', ascending=False)
    
    # Display the table
    st.table(display_df)
    
    # Model selection recommendations
    st.markdown("### Model Selection Guidance")
    st.markdown(
        """
        **When to choose each model type:**
        
        - **Random Forest**: Best all-around performer with balanced precision and recall. Good for most cases.
        - **Gradient Boosting**: Highest accuracy but may be more computationally intensive.
        - **SVM**: Works well with complex boundaries and medium-sized datasets.
        - **Logistic Regression**: When interpretability is important and the relationship is mostly linear.
        - **KNN**: Simple to understand and implement, works well with small datasets.
        - **Naive Bayes**: Very fast, works well with high-dimensional data like text.
        
        For this breast cancer classification task, tree-based models (Random Forest and Gradient Boosting) 
        generally perform best due to their ability to capture complex non-linear relationships in the data.
        """
    )

# Run the app
if __name__ == "__main__":
    main() 