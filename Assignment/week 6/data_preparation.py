#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preparation Script

This script extracts the Breast Cancer Wisconsin dataset from scikit-learn
and saves it in CSV format to the following locations:
- data/original: The complete original dataset
- data/processed: The training and test splits

Author: Professional Developer
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    """Main function to prepare dataset"""
    print("===== Preparing Breast Cancer Wisconsin Dataset =====")
    
    # Create directories if they don't exist
    os.makedirs("data/original", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Save the original dataset
    full_dataset = pd.concat([X, y], axis=1)
    full_dataset.to_csv("data/original/breast_cancer_dataset.csv", index=False)
    print("Original dataset saved to data/original/breast_cancer_dataset.csv")
    
    # Save feature names and target description
    with open("data/original/dataset_description.txt", "w") as f:
        f.write("Breast Cancer Wisconsin Dataset\n")
        f.write("==============================\n\n")
        f.write("Target Variable:\n")
        f.write("  0: malignant\n")
        f.write("  1: benign\n\n")
        f.write("Features:\n")
        for feature in data.feature_names:
            f.write(f"  - {feature}\n")
    print("Dataset description saved to data/original/dataset_description.txt")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Save train and test datasets
    train_df = pd.concat([pd.DataFrame(X_train, columns=data.feature_names), 
                          pd.Series(y_train, name='target')], axis=1)
    test_df = pd.concat([pd.DataFrame(X_test, columns=data.feature_names), 
                         pd.Series(y_test, name='target')], axis=1)
    
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("Training dataset saved to data/processed/train.csv")
    print("Testing dataset saved to data/processed/test.csv")
    
    # Create standardized versions of train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_scaled_df = pd.concat([pd.DataFrame(X_train_scaled, columns=data.feature_names), 
                                pd.Series(y_train, name='target')], axis=1)
    test_scaled_df = pd.concat([pd.DataFrame(X_test_scaled, columns=data.feature_names), 
                               pd.Series(y_test, name='target')], axis=1)
    
    train_scaled_df.to_csv("data/processed/train_scaled.csv", index=False)
    test_scaled_df.to_csv("data/processed/test_scaled.csv", index=False)
    print("Standardized training dataset saved to data/processed/train_scaled.csv")
    print("Standardized testing dataset saved to data/processed/test_scaled.csv")
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main() 