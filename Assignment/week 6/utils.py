import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib
import os

def load_data(use_preprocessed=True):
    """
    Load the breast cancer dataset from CSV files.
    
    Parameters:
        use_preprocessed: Whether to use the already preprocessed (scaled) data
    
    Returns:
        X_train, X_test, y_train, y_test: preprocessed train and test data
    """
    # Check if data files exist, if not create them
    if not os.path.exists("data/processed/train.csv"):
        print("Dataset files not found. Creating them now...")
        # Import and run the data preparation script
        import data_preparation
        data_preparation.main()
    
    if use_preprocessed:
        # Load preprocessed (scaled) data
        train_df = pd.read_csv("data/processed/train_scaled.csv")
        test_df = pd.read_csv("data/processed/test_scaled.csv")
    else:
        # Load raw split data
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")
    
    # Extract features and target
    y_train = train_df['target']
    X_train = train_df.drop('target', axis=1)
    
    y_test = test_df['target']
    X_test = test_df.drop('target', axis=1)
    
    # Print dataset information
    print(f"Dataset shape: {X_train.shape[0] + X_test.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    feature_names = X_train.columns
    print(f"Feature names: {feature_names.tolist()[:5]} ... (and {len(feature_names) - 5} more)")
    print(f"Target distribution (training):\n{y_train.value_counts()}")
    
    if not use_preprocessed:
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler for future use
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    # If using preprocessed data, return as is
    return X_train.values, X_test.values, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the performance of a trained model.
    
    Parameters:
        model: A trained scikit-learn model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for printing
        
    Returns:
        metrics_dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # For ROC AUC, we need probability predictions
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except (AttributeError, IndexError):
        roc_auc = None
    
    # Calculate metrics
    metrics_dict = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc
    }
    
    # Print detailed evaluation
    print(f"\n===== {model_name} Evaluation =====")
    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall: {metrics_dict['recall']:.4f}")
    print(f"F1 Score: {metrics_dict['f1']:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics_dict

def save_model(model, model_name):
    """
    Save a trained model to disk.
    
    Parameters:
        model: A trained scikit-learn model
        model_name: Name to save the model as
    """
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_name):
    """
    Load a trained model from disk.
    
    Parameters:
        model_name: Name of the saved model
        
    Returns:
        model: The loaded model
    """
    model_path = f"models/{model_name}.joblib"
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    return model 