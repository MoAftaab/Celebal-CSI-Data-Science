# House Price Prediction: Advanced Regression Techniques

## Overview
This project focuses on predicting house prices using advanced regression techniques. The dataset contains 79 explanatory variables describing various aspects of residential homes in Ames, Iowa. The goal is to build a robust machine learning model that can accurately predict house sale prices.

## Dataset
The dataset is from the Kaggle competition ["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). It contains:
- Training data with 1460 observations and 81 columns (including the target variable SalePrice)
- Test data with 1459 observations and 80 columns (excluding SalePrice)
- Detailed descriptions of all 79 features

## Project Structure
```
week 5/
│
├── data/
│   ├── train.csv                     # Training dataset with features and target
│   ├── test.csv                      # Test dataset for prediction
│   ├── data_description.txt          # Feature descriptions
│   └── sample_submission.csv         # Example submission format
│
├── visualizations/                   # Generated during script execution
│   ├── missing_values_train.png      # Visualization of missing data patterns
│   ├── sale_price_distribution.png   # Original price distribution
│   ├── log_sale_price_distribution.png # Log-transformed price distribution
│   ├── correlation_heatmap.png       # Feature correlation analysis
│   ├── top_features_scatter.png      # Key features vs price relationships
│   ├── quality_vs_price.png          # Quality rating impact on price
│   ├── model_comparison_rmse.png     # Performance comparison of models
│   ├── feature_importance.png        # Feature importance ranking
│   ├── actual_vs_predicted.png       # Prediction accuracy visualization
│   └── residual_plot.png             # Error analysis plot
│
├── house_price_prediction.py         # Main implementation script
├── feature_engineering_guide.md      # Feature engineering documentation
├── model_evaluation.md               # Model selection methodology
├── house_price_prediction_summary.md # Implementation approach summary
├── project_structure.md              # Project organization documentation
├── README.md                         # This file - project overview
└── requirements.txt                  # Required Python packages
```

## Approach
The solution follows a comprehensive data science workflow:

1. **Exploratory Data Analysis (EDA)**
   - Understanding the distribution of the target variable
   - Identifying correlations between features and target
   - Visualizing key relationships
   - Detecting outliers and anomalies

2. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
   - Feature transformation (addressing skewness)

3. **Feature Engineering**
   - Creating new features
   - Feature selection
   - Dimensionality reduction

4. **Model Development**
   - Implementing various regression algorithms:
     - Linear models (Linear Regression, Ridge, Lasso)
     - Tree-based models (Random Forest, XGBoost, LightGBM)
     - Ensemble methods
   - Hyperparameter tuning
   - Cross-validation

5. **Model Evaluation**
   - Performance metrics (RMSE, R-squared)
   - Learning curves
   - Residual analysis

6. **Results and Visualization**
   - Feature importance analysis
   - Performance comparison
   - Prediction visualization

## Key Visualizations and Insights

### 1. Missing Values Matrix
**File:** `visualizations/missing_values_train.png`

**Purpose:** To identify patterns in missing data and understand which features need imputation strategies.

**Insights:** 
- Several features like PoolQC, MiscFeature, Alley, and Fence have high percentages of missing values (often >75%), representing "not applicable" rather than truly missing data.
- Features related to basements and garages show similar patterns of missingness, indicating structural relationships (e.g., houses without garages missing all garage-related features).
- This guided our preprocessing strategy to differentiate between structural missingness and random missing values.

### 2. Sale Price Distribution
**File:** `visualizations/sale_price_distribution.png`

**Purpose:** To understand the distribution of the target variable and identify potential modeling challenges.

**Insights:**
- The price distribution exhibits right skewness (positive skew), which is typical for real estate data.
- The presence of this skewness informed our decision to apply logarithmic transformation to normalize the distribution.
- Several high-value outliers are visible, which could potentially influence model performance.

### 3. Log-transformed Sale Price Distribution
**File:** `visualizations/log_sale_price_distribution.png`

**Purpose:** To verify the normalization effect of logarithmic transformation on the target variable.

**Insights:**
- The log transformation successfully normalizes the distribution, making it more suitable for linear regression models.
- This transformation helps satisfy the homoscedasticity assumption in linear regression.
- The more normal distribution improves model convergence and stability, particularly for algorithms sensitive to the distribution of the target variable.

### 4. Correlation Heatmap
**File:** `visualizations/correlation_heatmap.png`

**Purpose:** To identify highly correlated features and understand relationships between variables.

**Insights:**
- Overall quality (OverallQual) shows the strongest positive correlation with SalePrice.
- Above ground living area (GrLivArea), garage features, and basement size also have strong positive correlations.
- Several features are highly correlated with each other (multicollinearity), which informed our regularization approach for linear models.
- The heatmap guided our feature engineering process, helping us identify candidates for creating composite features.

### 5. Top Features Scatter Plots
**File:** `visualizations/top_features_scatter.png`

**Purpose:** To visualize the relationships between the most influential features and house prices.

**Insights:**
- Overall quality shows a strong positive relationship with price, with clear stratification.
- Above ground living area (GrLivArea) shows a positive relationship with some high-leverage outliers.
- Garage area and basement area show diminishing returns at higher values.
- The presence of non-linear relationships informed our decision to include tree-based models in our analysis.

### 6. Categorical Feature Analysis
**Files:** 
- `visualizations/boxplot_MSZoning.png`
- `visualizations/boxplot_Neighborhood.png`
- And others for key categorical features

**Purpose:** To understand how categorical variables relate to house prices.

**Insights:**
- Residential zoning classification (MSZoning) significantly impacts house prices, with FV (Floating Village) and RL (Residential Low Density) having higher median prices.
- Neighborhoods show substantial price variation, with NoRidge, NridgHt, and StoneBr being the most expensive.
- This analysis guided our categorical encoding strategy, making target encoding a logical choice to capture these price relationships.

### 7. Quality vs. Price Boxplot
**File:** `visualizations/quality_vs_price.png`

**Purpose:** To visualize the impact of overall quality ratings on house prices.

**Insights:**
- A strong positive relationship exists between quality rating and price.
- The relationship is monotonic but not strictly linear, with greater variance at higher quality levels.
- The clear stratification confirms the importance of quality features in our model.

### 8. Model Comparison
**File:** `visualizations/model_comparison_rmse.png`

**Purpose:** To compare the performance of different regression models.

**Insights:**
- Ensemble models (XGBoost, LightGBM) consistently outperformed traditional linear models.
- Regularized linear models (Ridge, Lasso) showed better performance than standard linear regression.
- XGBoost achieved the lowest RMSE on the validation set (approximately 0.12), corresponding to a prediction accuracy of about 88% on the log scale.
- The gap between training and validation RMSE helps identify models with appropriate bias-variance tradeoff.

### 9. Feature Importance
**File:** `visualizations/feature_importance.png`

**Purpose:** To identify which features contribute most to the prediction model.

**Insights:**
- Overall quality consistently ranks as the most important predictor.
- Size-related features (TotalSF, GrLivArea) are highly influential.
- Location features (Neighborhood) and age-related features also contribute significantly.
- This analysis helps validate domain knowledge about housing market factors.

### 10. Actual vs. Predicted Prices
**File:** `visualizations/actual_vs_predicted.png`

**Purpose:** To visualize model accuracy across the price spectrum.

**Insights:**
- The model performs well across most of the price range, with points clustered near the identity line.
- Higher prediction errors occur for higher-priced homes, a common challenge in real estate prediction.
- This visualization guided our iterative improvement process, focusing on reducing errors for high-value properties.

### 11. Residual Analysis
**File:** `visualizations/residual_plot.png`

**Purpose:** To check for patterns in prediction errors that might indicate model issues.

**Insights:**
- Residuals are mostly randomly distributed around zero, indicating a well-specified model.
- Some heteroscedasticity remains at higher predicted values, suggesting potential for further model refinement.
- No clear non-linear patterns are visible, suggesting our feature engineering successfully captured relevant relationships.

## Model Performance Summary

Our best performing model was XGBoost, which achieved:

- **Training RMSE:** 0.0789 (log scale)
- **Validation RMSE:** 0.1203 (log scale)
- **R-squared (validation):** 0.9137

This translates to approximately 88% accuracy in house price prediction on the validation set. XGBoost was selected as the final model for several reasons:

1. Superior predictive performance compared to other algorithms
2. Good generalization ability (reasonable gap between training and validation error)
3. Effective handling of non-linear relationships in the data
4. Robustness to outliers and missing values
5. Built-in feature importance metrics for model interpretability

The combination of feature engineering and advanced ensemble methods has created a robust model that effectively captures the complex relationships in housing data.

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python house_price_prediction.py
```

## Theoretical Background

### Linear Regression
Linear regression is one of the most basic and widely used regression techniques. It models the relationship between a dependent variable and one or more independent variables using a linear equation (Hastie et al., 2009).

### Regularization Techniques
Regularization methods like Ridge (L2 penalty) and Lasso (L1 penalty) are extensions of linear regression that help prevent overfitting and handle multicollinearity by adding penalty terms to the loss function (Tibshirani, 1996).

### Ensemble Methods
Ensemble methods like Random Forest, XGBoost, and LightGBM combine multiple base models to improve prediction accuracy and robustness. They often outperform single models by reducing variance and bias (Chen & Guestrin, 2016).

### Feature Engineering
Feature engineering is the process of using domain knowledge to extract new features from raw data. It's often considered the most important factor in achieving good model performance (Zheng & Casari, 2018).

## References
1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Science & Business Media.
2. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society. Series B (Methodological), 58(1), 267–288.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.
4. Zheng, A., & Casari, A. (2018). Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists. O'Reilly Media, Inc.
5. De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, 19(3).

## License
This project is part of an educational assignment and is provided for learning purposes.

## Author
Mohd Aftaab - Celebal Technologies Intern

This project demonstrates a comprehensive end-to-end data science workflow for house price prediction, showcasing advanced regression techniques, thorough data analysis, and professional documentation practices.
