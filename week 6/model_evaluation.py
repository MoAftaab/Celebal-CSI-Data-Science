#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation and Hyperparameter Tuning

This script demonstrates the process of training multiple machine learning models,
evaluating their performance using various metrics, and optimizing their parameters
through systematic hyperparameter tuning.

Author: Professional Developer
"""

import numpy as np
import pandas as pd
import time
import joblib
import os
from sklearn.model_selection import (
    cross_val_score, 
    GridSearchCV, 
    RandomizedSearchCV,
    StratifiedKFold
)

# Import necessary models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Import custom functions
from utils import load_data, evaluate_model, save_model
from visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metrics_comparison,
    plot_hyperparameter_search_results,
    plot_feature_importance,
    plot_learning_curve
)


def train_base_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple base models.
    
    Parameters:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        
    Returns:
        models_dict: Dictionary of trained models
        metrics_list: List of performance metrics for each model
    """
    print("\n===== Training Base Models =====")
    
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }
    
    # Train and evaluate each model
    trained_models = {}
    metrics_list = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics["training_time"] = time.time() - start_time
        
        # Store trained model and metrics
        trained_models[name] = model
        metrics_list.append(metrics)
        
        # Generate confusion matrix plot
        plot_confusion_matrix(y_test, model.predict(X_test), name)
        
    return trained_models, metrics_list


def perform_hyperparameter_tuning(X_train, X_test, y_train, y_test, base_models):
    """
    Perform hyperparameter tuning on selected models.
    
    Parameters:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        base_models: Dictionary of base models
        
    Returns:
        tuned_models: Dictionary of tuned models
        tuned_metrics: List of performance metrics for tuned models
    """
    print("\n===== Performing Hyperparameter Tuning =====")
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Select models to tune based on base performance
    tuned_models = {}
    tuned_metrics = []
    
    # 1. Random Forest Tuning (using GridSearchCV)
    print("\nTuning Random Forest using GridSearchCV...")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    
    rf_grid.fit(X_train, y_train)
    
    print(f"Best params for Random Forest: {rf_grid.best_params_}")
    print(f"Best score for Random Forest: {rf_grid.best_score_:.4f}")
    
    # Save best model and metrics
    best_rf = rf_grid.best_estimator_
    tuned_models["Random Forest (Tuned)"] = best_rf
    tuned_metrics.append(evaluate_model(best_rf, X_test, y_test, "Random Forest (Tuned)"))
    
    # Plot hyperparameter tuning results
    plot_hyperparameter_search_results(rf_grid, "n_estimators", scoring="F1 Score")
    
    # 2. Gradient Boosting Tuning (using RandomizedSearchCV)
    print("\nTuning Gradient Boosting using RandomizedSearchCV...")
    gb_param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 4, 6]
    }
    
    gb_random = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_distributions=gb_param_dist,
        n_iter=10,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    
    gb_random.fit(X_train, y_train)
    
    print(f"Best params for Gradient Boosting: {gb_random.best_params_}")
    print(f"Best score for Gradient Boosting: {gb_random.best_score_:.4f}")
    
    # Save best model and metrics
    best_gb = gb_random.best_estimator_
    tuned_models["Gradient Boosting (Tuned)"] = best_gb
    tuned_metrics.append(evaluate_model(best_gb, X_test, y_test, "Gradient Boosting (Tuned)"))
    
    # 3. SVM Tuning
    print("\nTuning SVM using GridSearchCV...")
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly']
    }
    
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid=svm_param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    
    svm_grid.fit(X_train, y_train)
    
    print(f"Best params for SVM: {svm_grid.best_params_}")
    print(f"Best score for SVM: {svm_grid.best_score_:.4f}")
    
    # Save best model and metrics
    best_svm = svm_grid.best_estimator_
    tuned_models["SVM (Tuned)"] = best_svm
    tuned_metrics.append(evaluate_model(best_svm, X_test, y_test, "SVM (Tuned)"))
    
    return tuned_models, tuned_metrics


def analyze_best_model(best_model, X_train, X_test, y_train, y_test, feature_names, model_name):
    """
    Perform detailed analysis on the best model
    
    Parameters:
        best_model: The best performing model
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        feature_names: List of feature names
        model_name: Name of the model
    """
    print(f"\n===== Detailed Analysis of {model_name} =====")
    
    # Learning curve analysis
    print("Generating learning curve...")
    plot_learning_curve(best_model, X_train, y_train, model_name)
    
    # Feature importance analysis (if applicable)
    try:
        print("Analyzing feature importance...")
        importance_df = plot_feature_importance(best_model, feature_names)
        print("Top 10 important features:")
        print(importance_df.head(10))
    except (AttributeError, TypeError) as e:
        print(f"Feature importance not available for this model: {e}")
    
    # Save the best model
    save_model(best_model, "best_model")


def main():
    """Main function to orchestrate the model evaluation and tuning process"""
    print("===== Model Evaluation and Hyperparameter Tuning =====")
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data()
    
    # Get feature names from the breast cancer dataset
    from sklearn.datasets import load_breast_cancer
    feature_names = load_breast_cancer().feature_names
    
    # Train base models
    base_models, base_metrics = train_base_models(X_train, X_test, y_train, y_test)
    
    # Plot ROC curves for all base models
    print("\nPlotting ROC curves for base models...")
    plot_roc_curve(base_models, X_test, y_test)
    
    # Plot metrics comparison for base models
    print("Plotting metrics comparison for base models...")
    plot_metrics_comparison(base_metrics)
    
    # Perform hyperparameter tuning
    tuned_models, tuned_metrics = perform_hyperparameter_tuning(
        X_train, X_test, y_train, y_test, base_models
    )
    
    # Combine all models and metrics
    all_models = {**base_models, **tuned_models}
    all_metrics = base_metrics + tuned_metrics
    
    # Plot ROC curves for all models (base + tuned)
    print("\nPlotting ROC curves for all models...")
    plot_roc_curve(all_models, X_test, y_test)
    
    # Plot metrics comparison for all models
    print("Plotting metrics comparison for all models...")
    plot_metrics_comparison(all_metrics)
    
    # Find the best model based on F1 score
    best_model_metrics = max(all_metrics, key=lambda x: x["f1"])
    best_model_name = best_model_metrics["model_name"]
    best_model = all_models[best_model_name]
    
    print(f"\nBest model based on F1 score: {best_model_name}")
    print(f"F1 score: {best_model_metrics['f1']:.4f}")
    
    # Perform detailed analysis on the best model
    analyze_best_model(
        best_model, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        feature_names,
        best_model_name
    )
    
    print("\n===== Analysis Complete =====")
    print(f"Results saved in 'visualizations' directory")
    print(f"Best model saved as 'best_model.joblib' in 'models' directory")


if __name__ == "__main__":
    main() 