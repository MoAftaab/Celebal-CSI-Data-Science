import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def set_style():
    """Set the style for visualizations"""
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14

def save_figure(fig, filename):
    """Save a figure to the visualizations directory"""
    os.makedirs("visualizations", exist_ok=True)
    filepath = f"visualizations/{filename}"
    fig.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure saved to {filepath}")

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix for a model's predictions"""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        ax=ax
    )
    
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xticklabels(["Malignant (0)", "Benign (1)"])
    ax.set_yticklabels(["Malignant (0)", "Benign (1)"])
    
    save_figure(fig, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")

def plot_roc_curve(models_dict, X_test, y_test):
    """Plot ROC curves for multiple models"""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for model_name, model in models_dict.items():
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr, 
                tpr, 
                lw=2, 
                label=f"{model_name} (AUC = {roc_auc:.3f})"
            )
        except (AttributeError, IndexError):
            print(f"Could not generate ROC curve for {model_name}")
    
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right")
    
    save_figure(fig, "roc_curves_comparison.png")

def plot_metrics_comparison(metrics_list):
    """Plot comparison of metrics across models"""
    set_style()
    
    # Convert list of metric dictionaries to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.set_index("model_name")
    
    # Create bar plots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    metrics = ["accuracy", "precision", "recall", "f1"]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        sns.barplot(
            y=metrics_df.index, 
            x=metrics_df[metric], 
            palette="viridis",
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Model")
        ax.grid(True, axis="x")
        
        # Add value labels to bars
        for j, value in enumerate(metrics_df[metric]):
            ax.text(value + 0.01, j, f"{value:.4f}", va="center")
    
    plt.tight_layout()
    save_figure(fig, "metrics_comparison.png")
    
    # Create heatmap of all metrics
    fig, ax = plt.subplots(figsize=(14, len(metrics_df) * 0.8 + 2))
    sns.heatmap(
        metrics_df.iloc[:, :-1] if "roc_auc" in metrics_df.columns else metrics_df, 
        annot=True, 
        fmt=".4f", 
        cmap="viridis", 
        ax=ax
    )
    ax.set_title("Performance Metrics Across Models")
    
    save_figure(fig, "metrics_heatmap.png")

def plot_hyperparameter_search_results(grid_search, param_name, scoring="score"):
    """Plot the results of hyperparameter search"""
    set_style()
    
    results = grid_search.cv_results_
    param_values = results[f"param_{param_name}"].data
    
    # Check if parameter values are numeric
    try:
        param_values = [float(val) for val in param_values]
        numeric = True
    except (ValueError, TypeError):
        numeric = False
    
    fig, ax = plt.subplots()
    
    if numeric:
        # Line plot for numeric parameters
        ax.plot(
            param_values, 
            results["mean_test_score"], 
            marker="o", 
            linestyle="-"
        )
        ax.fill_between(
            param_values,
            results["mean_test_score"] - results["std_test_score"],
            results["mean_test_score"] + results["std_test_score"],
            alpha=0.2
        )
    else:
        # Bar plot for categorical parameters
        sns.barplot(
            x=param_values, 
            y=results["mean_test_score"],
            ax=ax
        )
        
    ax.set_xlabel(f"Value of {param_name}")
    ax.set_ylabel(f"Mean {scoring}")
    ax.set_title(f"{scoring.capitalize()} vs. {param_name}")
    
    if not numeric:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_figure(fig, f"hyperparameter_tuning_{param_name}.png")
    
    # Return the best parameter value
    best_idx = np.argmax(results["mean_test_score"])
    best_value = results[f"param_{param_name}"][best_idx]
    best_score = results["mean_test_score"][best_idx]
    
    return best_value, best_score

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance for a model"""
    set_style()
    
    # Check if model has feature_importances_ attribute
    if not hasattr(model, "feature_importances_"):
        print("Model doesn't provide feature importances")
        return
    
    # Get feature importances and create a DataFrame
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        "Feature": top_features,
        "Importance": top_importances
    })
    
    # Plot feature importances
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        y="Feature", 
        x="Importance", 
        data=importance_df.sort_values("Importance"),
        palette="viridis",
        ax=ax
    )
    
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    
    save_figure(fig, "feature_importance.png")
    
    return importance_df

def plot_learning_curve(model, X_train, y_train, model_name, cv=5):
    """Plot learning curve for a model"""
    from sklearn.model_selection import learning_curve
    
    set_style()
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=train_sizes,
        scoring="accuracy", n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots()
    
    ax.plot(train_sizes, train_mean, label="Training score", marker="o")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2
    )
    
    ax.plot(train_sizes, test_mean, label="Cross-validation score", marker="o")
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2
    )
    
    ax.set_title(f"Learning Curve - {model_name}")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Accuracy Score")
    ax.legend(loc="best")
    ax.grid(True)
    
    save_figure(fig, f"{model_name.replace(' ', '_').lower()}_learning_curve.png") 