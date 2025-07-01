import os
import sys
import streamlit as st

# Print debug information
st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())

# Check if the week 7 directory exists
if os.path.exists("week 7"):
    st.write("'week 7' directory exists")
    st.write("Files in week 7 directory:", os.listdir("week 7"))
    
    # Add the week 7 directory to the path
    week7_path = os.path.join(os.path.dirname(__file__), "week 7")
    sys.path.append(week7_path)
    
    # Run the week 7 app directly
    try:
        # Change to the week 7 directory to ensure imports work correctly
        original_dir = os.getcwd()
        os.chdir(week7_path)
        
        # Now import the modules after changing directory
        from utils import BreastCancerModel, plot_feature_importance, plot_prediction_gauge, get_feature_explanation, plot_model_comparison
        from visualizations import plot_model_performance, plot_feature_distributions, create_prediction_input_ui
        
        # Import other necessary libraries
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import plotly.express as px
        from sklearn.datasets import load_breast_cancer
        
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
                color: #4b4bff;
            }
            
            .model-selector {
                padding: 1rem;
                background-color: #f8f9fa;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
            }
            
            .model-comparison {
                padding: 1rem;
                background-color: #f0f2f6;
                border-radius: 0.5rem;
                margin-top: 2rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Import the functions from week 7/app.py
        # Note: Instead of using exec to run the app.py file, we'll directly import/implement the main functionality
        
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
        
        # Load app functions from week 7/app.py
        if os.path.exists(os.path.join(week7_path, "app.py")):
            with open(os.path.join(week7_path, "app.py"), "r") as f:
                app_content = f.read()
                # Extract the function definitions for main pages
                exec(app_content)
                
            # Run the main app
            main()
        else:
            st.error(f"Could not find app.py in {week7_path}")
        
        # Change back to original directory
        os.chdir(original_dir)
    except Exception as e:
        st.error(f"Error running app: {str(e)}")
        st.exception(e)
        
else:
    st.error("Could not find 'week 7' directory")
    st.write("Current directory:", os.getcwd())
    st.write("Available directories:", [d for d in os.listdir() if os.path.isdir(d)]) 