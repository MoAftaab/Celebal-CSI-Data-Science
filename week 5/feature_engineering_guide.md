# Feature Engineering Guide for House Price Prediction

## Introduction
Feature engineering is the process of using domain knowledge to extract or create features from raw data that make machine learning algorithms work more effectively. In the context of house price prediction, appropriate feature engineering can significantly improve model performance by capturing important relationships and patterns in the data.

## Key Feature Engineering Techniques Used

### 1. Total Area Features
**Created Features:**
- `TotalSF`: Sum of basement area and above ground living areas (`TotalBsmtSF + 1stFlrSF + 2ndFlrSF`)

**Rationale:**
The total area of a house is often more predictive of its price than individual area components. By combining basement and above-ground areas, we create a comprehensive size metric that correlates strongly with price (De Cock, 2011).

### 2. Bathroom Aggregation
**Created Features:**
- `TotalBathrooms`: Combined count of full and half bathrooms (`FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`)

**Rationale:**
The total number of bathrooms is a significant factor in house pricing. Half bathrooms are weighted as 0.5 to reflect their lower impact compared to full bathrooms (Zheng & Casari, 2018).

### 3. Binary Indicators
**Created Features:**
- `HasPool`: Whether the house has a pool
- `HasGarage`: Whether the house has a garage
- `HasFireplace`: Whether the house has a fireplace
- `HasBasement`: Whether the house has a basement
- `IsNew`: Whether the house is new (≤ 2 years old)

**Rationale:**
Binary indicators simplify certain attributes, focusing on presence/absence rather than detailed characteristics. This can reduce dimensionality while retaining predictive power for features where the mere presence significantly impacts price (Kuhn & Johnson, 2019).

### 4. Age-Related Features
**Created Features:**
- `HouseAge`: Years between built date and sold date (`YrSold - YearBuilt`)
- `YearsSinceRemodel`: Years since last remodel (`YrSold - YearRemodAdd`)

**Rationale:**
Age-related features capture the depreciation effect on housing values. Both the original age and time since last remodel affect perceived value (Goodman & Thibodeau, 1995).

### 5. Quality Encoding
**Transformation:**
- Mapping ordinal quality indicators to numerical scale (Ex: 5, Gd: 4, TA: 3, Fa: 2, Po: 1, NA: 0)

**Rationale:**
Converting ordinal categories to numeric values preserves the inherent ranking in quality assessments while making them usable in regression models (Hastie et al., 2009).

### 6. Target Encoding for Categorical Variables
**Transformation:**
- Replacing categorical variables with average target value for each category

**Rationale:**
Target encoding captures the relationship between categorical variables and the target variable directly. This is particularly useful for high-cardinality categorical variables like neighborhoods where one-hot encoding would create too many features (Micci-Barreca, 2001).

## Feature Transformation

### Log Transformation of Target Variable
**Transformation:**
- Applying log transformation to `SalePrice`

**Rationale:**
House prices typically follow a right-skewed distribution. Log transformation helps normalize this distribution, making it more suitable for linear models and improving prediction accuracy (Box & Cox, 1964).

### Robust Scaling
**Transformation:**
- Applying RobustScaler to all features

**Rationale:**
Real estate data often contains outliers (e.g., extremely large or luxurious houses). Robust scaling using medians and interquartile ranges reduces the influence of these outliers on model training (Rousseeuw & Hubert, 2011).

## Feature Selection Strategies

### Correlation Analysis
High correlation with target: Features with strong correlation to sale price (e.g., OverallQual, GrLivArea) were prioritized.

### Domain Knowledge
Based on real estate expertise: Features known to be important in property valuation (e.g., location, size, quality) were retained regardless of statistical measures.

### Remove Redundant Features
Highly correlated features: When two features had very high correlation, the more interpretable or complete feature was retained.

## References

1. Box, G. E. P., & Cox, D. R. (1964). An Analysis of Transformations. Journal of the Royal Statistical Society: Series B, 26(2), 211–243.
2. De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, 19(3).
3. Goodman, A. C., & Thibodeau, T. G. (1995). Age-related heteroskedasticity in hedonic house price equations. Journal of Housing Research, 6(1), 25-42.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Science & Business Media.
5. Kuhn, M., & Johnson, K. (2019). Feature Engineering and Selection: A Practical Approach for Predictive Models. Chapman and Hall/CRC.
6. Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. ACM SIGKDD Explorations Newsletter, 3(1), 27-32.
7. Rousseeuw, P.J., & Hubert, M. (2011). Robust statistics for outlier detection. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 1(1), 73-79.
8. Zheng, A., & Casari, A. (2018). Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists. O'Reilly Media, Inc. 