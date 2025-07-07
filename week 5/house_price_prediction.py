#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
House Price Prediction using Advanced Regression Techniques

This script implements a comprehensive solution for predicting house prices
using the Ames Housing dataset. It includes data exploration, preprocessing,
feature engineering, model training, evaluation, and visualization.

Dataset: Ames Housing dataset - Kaggle competition
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
import missingno as msno
import joblib
import warnings
import category_encoders as ce

# Set visualization style
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('visualizations', exist_ok=True)

class HousePricePredictor:
    """
    A class for predicting house prices using advanced regression techniques.
    
    This class encapsulates the entire workflow from data loading to model evaluation.
    """
    
    def __init__(self, train_path='data/train.csv', test_path='data/test.csv'):
        """
        Initialize the HousePricePredictor with data paths.
        
        Parameters:
        -----------
        train_path : str
            Path to the training data CSV file
        test_path : str
            Path to the test data CSV file
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.X_val = None
        self.models = {}
        self.predictions = {}
        
    def load_data(self):
        """Load training and test datasets"""
        try:
            self.train_data = pd.read_csv(self.train_path)
            self.test_data = pd.read_csv(self.test_path)
            print(f"Data loaded successfully.")
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """
        Perform exploratory data analysis and generate visualizations.
        """
        if self.train_data is None:
            print("Data not loaded. Please load data first.")
            return
        
        print("\n=== Exploratory Data Analysis ===")
        
        # Basic information about the dataset
        print("\n--- Dataset Information ---")
        print(self.train_data.info())
        
        # Summary statistics
        print("\n--- Summary Statistics ---")
        print(self.train_data.describe().T)
        
        # Check for missing values
        print("\n--- Missing Values ---")
        missing_train = self.train_data.isnull().sum()[self.train_data.isnull().sum() > 0].sort_values(ascending=False)
        print(f"Training data missing values:\n{missing_train}")
        
        missing_test = self.test_data.isnull().sum()[self.test_data.isnull().sum() > 0].sort_values(ascending=False)
        print(f"\nTest data missing values:\n{missing_test}")
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        msno.matrix(self.train_data)
        plt.title('Missing Values in Training Data')
        plt.savefig('visualizations/missing_values_train.png', bbox_inches='tight')
        
        # Target variable distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.train_data['SalePrice'], kde=True)
        plt.title('Sale Price Distribution')
        plt.savefig('visualizations/sale_price_distribution.png', bbox_inches='tight')
        
        # Log-transformed target variable
        plt.figure(figsize=(10, 6))
        sns.histplot(np.log1p(self.train_data['SalePrice']), kde=True)
        plt.title('Log-transformed Sale Price Distribution')
        plt.savefig('visualizations/log_sale_price_distribution.png', bbox_inches='tight')
        
        # Correlation with SalePrice
        plt.figure(figsize=(12, 10))
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
        correlation = self.train_data[numeric_cols].corr()
        top_corr_features = correlation['SalePrice'].sort_values(ascending=False)[:15]
        print("\n--- Top Features Correlated with SalePrice ---")
        print(top_corr_features)
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation.iloc[:15, :15], annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap (Top 15 Features)')
        plt.savefig('visualizations/correlation_heatmap.png', bbox_inches='tight')
        
        # Scatter plots for top correlated features
        top_features = top_corr_features.index.tolist()[1:6]  # Exclude SalePrice itself
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features):
            plt.subplot(2, 3, i+1)
            sns.scatterplot(x=self.train_data[feature], y=self.train_data['SalePrice'])
            plt.title(f'{feature} vs SalePrice')
            plt.tight_layout()
        plt.savefig('visualizations/top_features_scatter.png', bbox_inches='tight')
        
        # Categorical features analysis
        cat_cols = self.train_data.select_dtypes(include=['object']).columns
        for col in cat_cols[:5]:  # Limit to first 5 categorical features
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col, y='SalePrice', data=self.train_data)
            plt.xticks(rotation=45)
            plt.title(f'SalePrice by {col}')
            plt.tight_layout()
            plt.savefig(f'visualizations/boxplot_{col}.png', bbox_inches='tight')
            
        # Overall quality vs SalePrice
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='OverallQual', y='SalePrice', data=self.train_data)
        plt.title('SalePrice by Overall Quality')
        plt.savefig('visualizations/quality_vs_price.png', bbox_inches='tight')
        
        print("\nEDA completed. Visualizations saved in 'visualizations' directory.")
        
    def preprocess_data(self):
        """
        Preprocess the data, including handling missing values, encoding categorical variables,
        and feature transformation.
        """
        if self.train_data is None or self.test_data is None:
            print("Data not loaded. Please load data first.")
            return
        
        print("\n=== Data Preprocessing ===")
        
        # Combine datasets for consistent preprocessing
        self.train_data['is_train'] = 1
        self.test_data['is_train'] = 0
        
        # Save the target variable before combining
        self.y = np.log1p(self.train_data['SalePrice'])
        
        # Combine train and test for preprocessing
        combined_data = pd.concat([self.train_data.drop('SalePrice', axis=1), 
                                  self.test_data], axis=0)
        
        # Handle missing values
        print("\n--- Handling Missing Values ---")
        
        # Identify columns with high percentage of missing values
        missing_percent = combined_data.isnull().sum() / len(combined_data) * 100
        high_missing = missing_percent[missing_percent > 75].index.tolist()
        print(f"Features with >75% missing values (to be dropped): {high_missing}")
        
        # Drop features with high missing values
        combined_data.drop(high_missing, axis=1, inplace=True)
        
        # Fill missing values based on data types
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        categorical_cols = combined_data.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if combined_data[col].isnull().sum() > 0:
                combined_data[col].fillna(combined_data[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if combined_data[col].isnull().sum() > 0:
                combined_data[col].fillna(combined_data[col].mode()[0], inplace=True)
        
        print("Missing values handled.")
        
        # Feature Engineering
        print("\n--- Feature Engineering ---")
        
        # Create new features
        combined_data['TotalSF'] = combined_data['TotalBsmtSF'] + combined_data['1stFlrSF'] + combined_data['2ndFlrSF']
        combined_data['TotalBathrooms'] = (combined_data['FullBath'] + 0.5 * combined_data['HalfBath'] + 
                                         combined_data['BsmtFullBath'] + 0.5 * combined_data['BsmtHalfBath'])
        combined_data['HasPool'] = combined_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        combined_data['HasGarage'] = combined_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        combined_data['HasFireplace'] = combined_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
        combined_data['HasBasement'] = combined_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        combined_data['HouseAge'] = combined_data['YrSold'] - combined_data['YearBuilt']
        combined_data['IsNew'] = combined_data['HouseAge'].apply(lambda x: 1 if x <= 2 else 0)
        combined_data['YearsSinceRemodel'] = combined_data['YrSold'] - combined_data['YearRemodAdd']
        
        # Encode ordinal features
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        
        for col in quality_cols:
            if col in combined_data.columns:
                combined_data[col] = combined_data[col].map(quality_map).fillna(0)
        
        # Handle categorical features using target encoding
        print("\n--- Encoding Categorical Features ---")
        
        # Identify categorical columns after previous transformations
        categorical_cols = combined_data.select_dtypes(include=['object']).columns
        
        # Target encoding for categorical variables
        encoder = ce.TargetEncoder()
        
        # Split back to train and test
        train_data = combined_data[combined_data['is_train'] == 1].drop('is_train', axis=1)
        test_data = combined_data[combined_data['is_train'] == 0].drop(['is_train'], axis=1)
        
        # Apply target encoding only to categorical columns
        if len(categorical_cols) > 0:
            train_data_encoded = train_data.copy()
            test_data_encoded = test_data.copy()
            
            # Create a dummy target for the test set (will not be used)
            dummy_target = np.zeros(len(test_data))
            
            # Fit on train and transform both train and test
            for col in categorical_cols:
                if col in train_data.columns and col in test_data.columns:
                    encoder.fit(train_data[col], self.y)
                    train_data_encoded[col] = encoder.transform(train_data[col])
                    test_data_encoded[col] = encoder.transform(test_data[col])
            
            train_data = train_data_encoded
            test_data = test_data_encoded
            
            print(f"Encoded {len(categorical_cols)} categorical features.")
        
        # Prepare X_train and X_test
        self.X_train = train_data
        self.X_test = test_data
        
        # Feature scaling
        print("\n--- Feature Scaling ---")
        scaler = RobustScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        # Split training data for validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y, test_size=0.2, random_state=42
        )
        
        print("Data preprocessing completed.")
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        
    def train_models(self):
        """
        Train various regression models and evaluate their performance.
        """
        if self.X_train is None or self.y_train is None:
            print("Data not preprocessed. Please preprocess data first.")
            return
        
        print("\n=== Model Training ===")
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.001),
            'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions on validation set
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            
            train_r2 = r2_score(self.y_train, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)
            
            print(f"{name} - Training RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
            print(f"{name} - Training R²: {train_r2:.4f}, Validation R²: {val_r2:.4f}")
            
            # Save model and predictions
            self.models[name] = model
            self.predictions[name] = {
                'train_pred': train_pred,
                'val_pred': val_pred,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2
            }
            
        # Identify best model
        best_model_name = min(self.predictions.items(), key=lambda x: x[1]['val_rmse'])[0]
        print(f"\nBest model based on validation RMSE: {best_model_name}")
        print(f"Validation RMSE: {self.predictions[best_model_name]['val_rmse']:.4f}")
        print(f"Validation R²: {self.predictions[best_model_name]['val_r2']:.4f}")
        
        # Save best model
        joblib.dump(self.models[best_model_name], 'best_model.joblib')
        print("Best model saved as 'best_model.joblib'")
        
        return best_model_name
        
    def visualize_results(self, best_model_name):
        """
        Visualize model results and feature importance.
        
        Parameters:
        -----------
        best_model_name : str
            Name of the best performing model
        """
        if not self.predictions:
            print("No model predictions available. Please train models first.")
            return
        
        print("\n=== Results Visualization ===")
        
        # Model comparison
        plt.figure(figsize=(12, 6))
        val_rmse = [self.predictions[model]['val_rmse'] for model in self.predictions]
        train_rmse = [self.predictions[model]['train_rmse'] for model in self.predictions]
        
        x = np.arange(len(self.predictions))
        width = 0.35
        
        plt.bar(x - width/2, train_rmse, width, label='Training RMSE')
        plt.bar(x + width/2, val_rmse, width, label='Validation RMSE')
        
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Model Comparison - RMSE')
        plt.xticks(x, list(self.predictions.keys()), rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison_rmse.png', bbox_inches='tight')
        
        # Feature importance for the best model
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
            plt.title(f'Feature Importance - {best_model_name}')
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', bbox_inches='tight')
            
            print("\nTop 15 Important Features:")
            print(feature_importance.head(15))
        
        # Actual vs Predicted
        plt.figure(figsize=(10, 8))
        best_val_pred = self.predictions[best_model_name]['val_pred']
        
        plt.scatter(np.expm1(self.y_val), np.expm1(best_val_pred), alpha=0.5)
        plt.plot([0, max(np.expm1(self.y_val))], [0, max(np.expm1(self.y_val))], 'r--')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'Actual vs Predicted - {best_model_name}')
        plt.tight_layout()
        plt.savefig('visualizations/actual_vs_predicted.png', bbox_inches='tight')
        
        # Residual Plot
        plt.figure(figsize=(10, 6))
        residuals = self.y_val - best_val_pred
        plt.scatter(np.expm1(best_val_pred), residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.savefig('visualizations/residual_plot.png', bbox_inches='tight')
        
        print("Results visualization completed. Plots saved in 'visualizations' directory.")
        
    def make_predictions(self, best_model_name):
        """
        Make predictions on the test set and create submission file.
        
        Parameters:
        -----------
        best_model_name : str
            Name of the best performing model
        """
        if best_model_name not in self.models:
            print(f"Model {best_model_name} not found.")
            return
        
        print("\n=== Making Predictions on Test Set ===")
        
        # Make predictions
        best_model = self.models[best_model_name]
        test_preds = best_model.predict(self.X_test)
        
        # Transform predictions back to original scale
        test_preds_original = np.expm1(test_preds)
        
        # Create submission file
        submission = pd.DataFrame({
            'Id': self.test_data.Id,
            'SalePrice': test_preds_original
        })
        
        submission.to_csv('submission.csv', index=False)
        print("Predictions completed. Submission file created: 'submission.csv'")
        
    def run_pipeline(self):
        """
        Run the complete pipeline from data loading to predictions.
        """
        print("=== House Price Prediction Pipeline ===")
        
        # Load data
        if not self.load_data():
            return
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        best_model_name = self.train_models()
        
        # Visualize results
        self.visualize_results(best_model_name)
        
        # Make predictions
        self.make_predictions(best_model_name)
        
        print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    # Define paths
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    # Initialize and run the pipeline
    predictor = HousePricePredictor(train_path, test_path)
    predictor.run_pipeline() 