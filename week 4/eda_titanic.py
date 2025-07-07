import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def eda_titanic():
    """
    Performs an in-depth Exploratory Data Analysis (EDA) on the Titanic dataset.

    This function loads the Titanic dataset, performs data cleaning and feature
    engineering, and generates a series of visualizations to understand data
    distributions, relationships between variables, and factors influencing
    survival. The visualizations are saved to the 'visualizations' directory.
    """

    # --- 1. Setup --- 
    print("Starting Titanic EDA...")
    # Create a directory to save visualizations
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Set plot style
    sns.set(style='whitegrid')
    plt.style.use('seaborn-v0_8-talk')

    # --- 2. Load Data ---
    print("Loading dataset...")
    df = sns.load_dataset('titanic')

    # --- 3. Initial Data Inspection ---
    print("\nInitial Data Info:")
    df.info()

    print("\n\nSummary Statistics:")
    print(df.describe())

    print("\n\nMissing Values:")
    print(df.isnull().sum())

    # --- 4. Data Cleaning & Preprocessing ---
    print("\nCleaning data...")
    # Fill missing 'age' values with the median
    df['age'].fillna(df['age'].median(), inplace=True)

    # Fill missing 'embarked' and 'embark_town' with the mode
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

    # Drop the 'deck' column due to too many missing values
    df.drop('deck', axis=1, inplace=True)

    print("\nData cleaned. Missing values handled.")

    # --- 5. Feature Engineering ---
    print("Performing feature engineering...")
    # Create FamilySize feature
    df['family_size'] = df['sibsp'] + df['parch'] + 1

    # Create IsAlone feature
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    print("Feature engineering complete.")

    # --- 6. Visualization Generation ---
    print("Generating visualizations...")

    # Plot 1: Survival Count
    plt.figure(figsize=(8, 6))
    sns.countplot(x='survived', data=df, palette='viridis')
    plt.title('Survival Count (0 = No, 1 = Yes)')
    plt.savefig('visualizations/1_survival_count.png')
    plt.close()

    # Plot 2: Survival by Sex
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sex', hue='survived', data=df, palette='plasma')
    plt.title('Survival Count by Sex')
    plt.savefig('visualizations/2_survival_by_sex.png')
    plt.close()

    # Plot 3: Survival by Pclass
    plt.figure(figsize=(8, 6))
    sns.countplot(x='pclass', hue='survived', data=df, palette='magma')
    plt.title('Survival Count by Passenger Class')
    plt.savefig('visualizations/3_survival_by_pclass.png')
    plt.close()

    # Plot 4: Age Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title('Age Distribution of Passengers')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('visualizations/4_age_distribution.png')
    plt.close()

    # Plot 5: Age Distribution by Survival
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df, x='age', hue='survived', multiple='stack', palette='coolwarm', fill=True)
    plt.title('Age Distribution by Survival Status')
    plt.xlabel('Age')
    plt.savefig('visualizations/5_age_distribution_by_survival.png')
    plt.close()

    # Plot 6: Fare Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['fare'], bins=40, kde=True, color='green')
    plt.title('Fare Distribution')
    plt.xlabel('Fare')
    plt.savefig('visualizations/6_fare_distribution.png')
    plt.close()

    # Plot 7: Survival by Family Size
    plt.figure(figsize=(12, 7))
    sns.countplot(x='family_size', hue='survived', data=df, palette='cubehelix')
    plt.title('Survival Count by Family Size')
    plt.savefig('visualizations/7_survival_by_family_size.png')
    plt.close()

    # Plot 8: Box plot of Age by Pclass and Sex
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='pclass', y='age', hue='sex', data=df, palette='seismic')
    plt.title('Age Distribution by Pclass and Sex')
    plt.savefig('visualizations/8_age_by_pclass_sex.png')
    plt.close()

    # Plot 9: Correlation Heatmap
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig('visualizations/9_correlation_heatmap.png')
    plt.close()

    # Plot 10: Pairplot
    # Using a sample to keep the plot readable and generation fast
    pairplot_df = df[['survived', 'pclass', 'age', 'fare', 'family_size']].sample(n=300, random_state=42)
    sns.pairplot(pairplot_df, hue='survived', palette='husl', diag_kind='kde')
    plt.suptitle('Pairplot of Key Numerical Features by Survival', y=1.02)
    plt.savefig('visualizations/10_pairplot.png')
    plt.close()

    print("\nAll visualizations have been generated and saved in the 'visualizations' directory.")
    print("EDA complete.")

if __name__ == '__main__':
    eda_titanic()
