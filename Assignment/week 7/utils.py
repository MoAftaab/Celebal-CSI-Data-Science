import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

class BreastCancerModel:
    """Class to handle the breast cancer prediction model operations"""
    
    def __init__(self):
        """Initialize the model by loading the saved model and scaler"""
        # Define paths relative to the parent directory (week 6)
        self.week7_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.join(self.week7_dir, "..")
        self.model_path = os.path.join(parent_dir, "week 6", "models", "best_model.joblib")
        self.scaler_path = os.path.join(parent_dir, "week 6", "models", "scaler.joblib")
        
        # Try to load model from week 6 first, if not available, use a default model
        try:
            # Load the trained model and scaler
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except (FileNotFoundError, OSError):
            # If model not found, create a default one
            print("Model files not found. Creating default models...")
            self.model = self.train_default_model()
            self.scaler = StandardScaler()
        
        # Get feature names from the breast cancer dataset
        self.feature_names = load_breast_cancer().feature_names
        self.dataset = load_breast_cancer()
        
        # Store all available models
        self.models = {
            "Random Forest": self.model,  # Default loaded model
        }
        
        self.model_name = "Random Forest"  # Default model name
        
        # Create feature info dictionary
        self.feature_info = {}
        for i, name in enumerate(self.feature_names):
            self.feature_info[name] = {
                'min': self.dataset.data[:, i].min(),
                'max': self.dataset.data[:, i].max(),
                'mean': self.dataset.data[:, i].mean(),
                'std': self.dataset.data[:, i].std()
            }
            
        # Store the dataset for permutation importance
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for training and evaluation"""
        # Load dataset
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store the scaled data for later use
        self.X_train = X_train_scaled
        self.y_train = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.scaler = scaler
    
    def train_default_model(self):
        """Train a default model if the saved model is not available"""
        # Load dataset
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Train a Random Forest model
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Save the model to week 7 models directory
        joblib.dump(model, os.path.join(self.week7_dir, "models", "random_forest.joblib"))
        
        return model
    
    def train_additional_models(self):
        """Train additional models for comparison"""
        # Create a dictionary to store models and their metrics
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),  # Increased iterations
            "SVM": SVC(probability=True, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }
        
        # Train all models
        model_metrics = {}
        for name, model in models.items():
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Get predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
            }
            
            # Save model
            model_path = os.path.join(self.week7_dir, "models", f"{name.lower().replace(' ', '_')}.joblib")
            joblib.dump(model, model_path)
            
            # Store model and metrics
            self.models[name] = model
            model_metrics[name] = metrics
        
        return model_metrics
    
    def set_model(self, model_name):
        """
        Set the active model for predictions
        
        Parameters:
            model_name: Name of the model to use
        """
        if model_name in self.models:
            self.model = self.models[model_name]
            self.model_name = model_name
            return True
        
        # If model not already loaded, try to load it from file
        try:
            model_path = os.path.join(self.week7_dir, "models", f"{model_name.lower().replace(' ', '_')}.joblib")
            self.model = joblib.load(model_path)
            self.models[model_name] = self.model
            self.model_name = model_name
            return True
        except (FileNotFoundError, OSError):
            # If model file doesn't exist, train all models
            metrics = self.train_additional_models()
            self.model = self.models[model_name]
            self.model_name = model_name
            return True
    
    def predict(self, input_data):
        """
        Make a prediction using the trained model
        
        Parameters:
            input_data: DataFrame or array with patient features
            
        Returns:
            prediction: 0 (malignant) or 1 (benign)
            probability: Probability of the prediction
        """
        # Convert input to numpy array if it's a DataFrame
        if isinstance(input_data, pd.DataFrame):
            input_array = input_data.values
        else:
            input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input data
        scaled_input = self.scaler.transform(input_array)
        
        # Make prediction
        prediction = self.model.predict(scaled_input)[0]
        
        # Get prediction probability
        probabilities = self.model.predict_proba(scaled_input)[0]
        probability = probabilities[1] if prediction == 1 else probabilities[0]
        
        return prediction, probability
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance for any model type
        
        Parameters:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance values
        """
        # First try native feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            })
            
            # Sort by importance and get top N
            return importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        elif hasattr(self.model, 'coef_'):
            # For linear models like Logistic Regression
            coefficients = self.model.coef_[0]
            
            # Take absolute values for importance
            importances = np.abs(coefficients)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            })
            
            # Sort by importance and get top N
            return importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        else:
            # For models without native feature importance, use permutation importance
            try:
                # Calculate permutation importance
                result = permutation_importance(
                    self.model, self.X_test, self.y_test, 
                    n_repeats=10, random_state=42, n_jobs=-1
                )
                
                # Extract importance values
                importances = result.importances_mean
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importances
                })
                
                # Sort by importance and get top N
                return importance_df.sort_values('Importance', ascending=False).head(top_n)
            except:
                # If permutation importance fails, return None
                return None
    
    def get_all_models_metrics(self):
        """
        Get performance metrics for all trained models
        
        Returns:
            DataFrame with model metrics
        """
        # Ensure all models are trained
        if len(self.models) <= 1:
            self.train_additional_models()
        
        # Get metrics for each model
        metrics_list = []
        
        for name, model in self.models.items():
            # Get predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            metrics = {
                'model': name,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
            }
            
            metrics_list.append(metrics)
        
        # Convert to DataFrame
        return pd.DataFrame(metrics_list)
    
    def get_sample_data(self, n_samples=5):
        """
        Get random samples from the original dataset
        
        Parameters:
            n_samples: Number of samples to return
            
        Returns:
            DataFrame with sample data
        """
        # Get random indices
        random_indices = np.random.choice(
            self.dataset.data.shape[0], 
            size=n_samples, 
            replace=False
        )
        
        # Create DataFrame with selected samples
        sample_df = pd.DataFrame(
            self.dataset.data[random_indices], 
            columns=self.feature_names
        )
        
        # Add target column
        sample_df['target'] = self.dataset.target[random_indices]
        
        return sample_df
    
    def get_feature_ranges(self):
        """
        Get min, max values for each feature
        
        Returns:
            Dictionary with feature ranges
        """
        return self.feature_info

# Helper functions for visualizations
def plot_feature_importance(importance_df):
    """Create a plotly bar chart of feature importance"""
    if importance_df is None:
        # Return empty figure if no feature importance available
        fig = go.Figure()
        fig.update_layout(
            title="Feature importance not available for this model type",
            height=500
        )
        return fig
    
    fig = px.bar(
        importance_df.sort_values('Importance'),
        y='Feature',
        x='Importance',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=500,
        xaxis_title='Importance',
        yaxis_title='Feature',
        coloraxis_showscale=False
    )
    
    return fig

def plot_prediction_gauge(probability, prediction):
    """Create a gauge chart showing the prediction probability"""
    label = "Benign" if prediction == 1 else "Malignant"
    color = "#86efac" if prediction == 1 else "#b91c1c"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': f"Prediction: {label}"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ]
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(t=60, b=0, l=20, r=20)
    )
    
    return fig

def plot_model_comparison(metrics_df):
    """Create a plotly bar chart comparing model metrics"""
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(
        metrics_df, 
        id_vars=['model'], 
        value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create the figure
    fig = px.bar(
        melted_df,
        x='model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        labels={'model': 'Model', 'Value': 'Score', 'Metric': 'Metric'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        height=600,
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis=dict(range=[0.5, 1.0]),  # Adjusted range to ensure all metrics are visible
        legend_title='Metric',
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def get_feature_explanation(feature_name, value, importance_df):
    """Get explanation for a feature's contribution to the prediction"""
    if importance_df is None or feature_name not in importance_df['Feature'].values:
        return ""
    
    importance = importance_df.loc[importance_df['Feature'] == feature_name, 'Importance'].values
    if len(importance) == 0:
        return ""
    
    importance = importance[0]
    feature_rank = importance_df['Feature'].tolist().index(feature_name) + 1
    
    explanation = f"This feature ranks #{feature_rank} in importance (score: {importance:.4f}). "
    
    return explanation 