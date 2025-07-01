import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import sys
from sklearn.datasets import load_breast_cancer

# Add parent directory to path to import modules from week 6
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from utils import BreastCancerModel, plot_feature_importance, plot_prediction_gauge, get_feature_explanation
from visualizations import plot_model_performance, plot_feature_distributions, create_prediction_input_ui

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        color: #4b4bff;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .description {
        font-size: 1.1rem;
        text-align: justify;
        margin-bottom: 2rem;
    }
    
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .prediction-benign {
        color: green;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .prediction-malignant {
        color: red;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .accuracy-metric {
        font-size: 1.2rem;
        font-weight: bold;
        color: #4b4bff;
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
    try:
        # Try to import the load_data function
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week 6"))
        from week6.utils import load_data
        X_train, X_test, y_train, y_test = load_data()
        return X_test, y_test
    except ImportError:
        # If import fails, load breast cancer dataset directly
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
        This web application uses a machine learning model to predict whether a breast mass is benign or malignant
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
    page = st.sidebar.radio("Go to", ["Prediction", "Model Performance", "Data Exploration"])
    
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app demonstrates the deployment of a machine learning model using Streamlit.
        
        **Dataset:** Wisconsin Breast Cancer Dataset
        
        **Model:** Random Forest Classifier
        
        **Task:** Binary Classification (Malignant vs. Benign)
        """
    )
    
    # Content based on selected page
    if page == "Prediction":
        prediction_page(model)
    elif page == "Model Performance":
        performance_page(model)
    else:
        exploration_page(model)

def prediction_page(model):
    st.markdown("<h2 class='sub-header'>Cancer Prediction</h2>", unsafe_allow_html=True)
    
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
                    <p>The tumor is predicted to be: <span class='{prediction_class}'>{prediction_label}</span> with {probability:.2%} confidence</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display gauge chart
            st.plotly_chart(plot_prediction_gauge(probability, prediction))
            
            # Display feature importance and explanation
            st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
            importance_df = model.get_feature_importance()
            
            if importance_df is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(plot_feature_importance(importance_df))
                
                with col2:
                    st.markdown("<h4>Top Features Explanation</h4>", unsafe_allow_html=True)
                    for idx, row in importance_df.head(5).iterrows():
                        feature = row['Feature']
                        if feature in input_data.columns:
                            value = input_data[feature].values[0]
                            explanation = get_feature_explanation(feature, value, importance_df)
                            st.markdown(f"**{feature}**: {value:.6f}<br>{explanation}", unsafe_allow_html=True)
                            st.divider()

def performance_page(model):
    st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
    
    # Load the dataset
    dataset = load_breast_cancer()
    
    # Create model metrics section
    st.markdown("### Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; text-align:center;">
                <h4>Accuracy</h4>
                <p class="accuracy-metric">97.2%</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; text-align:center;">
                <h4>Precision</h4>
                <p class="accuracy-metric">98.1%</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; text-align:center;">
                <h4>Recall</h4>
                <p class="accuracy-metric">97.3%</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            """
            <div style="border:1px solid #ccc; border-radius:5px; padding:10px; text-align:center;">
                <h4>F1 Score</h4>
                <p class="accuracy-metric">97.7%</p>
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
        st.info("Displaying pre-computed performance metrics instead.")
        
        # Show saved visualizations
        st.image("../week 6/visualizations/random_forest_confusion_matrix.png", caption="Confusion Matrix")
        st.image("../week 6/visualizations/roc_curves_comparison.png", caption="ROC Curve")
    
    # Model details
    with st.expander("Model Architecture Details"):
        st.markdown(
            """
            ### Random Forest Classifier
            
            The breast cancer classification model uses a **Random Forest Classifier** with the following hyperparameters:
            
            - **n_estimators**: 200 (Number of trees in the forest)
            - **max_depth**: 10 (Maximum depth of each tree)
            - **min_samples_split**: 5 (Minimum samples required to split an internal node)
            - **min_samples_leaf**: 2 (Minimum samples required to be at a leaf node)
            - **criterion**: 'gini' (Function to measure the quality of a split)
            
            The model was trained on 75% of the Wisconsin Breast Cancer dataset and tested on the remaining 25%.
            All features were standardized using StandardScaler before training.
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

# Run the app
if __name__ == "__main__":
    main() 