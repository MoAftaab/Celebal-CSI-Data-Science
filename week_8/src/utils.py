import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a given path
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Loaded pandas DataFrame
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing on the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            # Fill missing values with mode
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Unknown")
            # Convert object columns to string for better compatibility with Streamlit
            processed_df[col] = processed_df[col].astype(str)
        else:
            # Fill missing numerical values with median
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    # Ensure boolean columns are properly typed
    for col in processed_df.columns:
        if set(processed_df[col].unique()).issubset({'Y', 'N', 'Yes', 'No', 'yes', 'no', 'True', 'False', 'true', 'false'}):
            # Map to boolean values
            bool_map = {'Y': True, 'N': False, 'Yes': True, 'No': False, 
                       'yes': True, 'no': False, 'True': True, 'False': False, 
                       'true': True, 'false': False}
            processed_df[col] = processed_df[col].map(lambda x: bool_map.get(x, x))
    
    return processed_df

def generate_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an overview of the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing dataset overview information
    """
    overview = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'column_types': dict(df.dtypes),
        'missing_values': df.isnull().sum().to_dict(),
        'descriptive_stats': df.describe().to_dict(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'numerical_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    }
    return overview

def create_loan_approval_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a summary of loan approval statistics
    
    Args:
        df: Input DataFrame with loan data
        
    Returns:
        Dictionary containing loan approval summary
    """
    try:
        # Check if the DataFrame has the necessary columns
        if 'Loan_Status' not in df.columns:
            return {"error": "Loan_Status column not found in the dataset"}
        
        approval_counts = df['Loan_Status'].value_counts().to_dict()
        approval_rate = (df['Loan_Status'] == 'Y').mean() * 100
        
        # Gender-based approval rate
        if 'Gender' in df.columns:
            gender_approval = df.groupby('Gender')['Loan_Status'].apply(
                lambda x: (x == 'Y').mean() * 100).to_dict()
        else:
            gender_approval = {"error": "Gender column not found"}
        
        # Credit History based approval rate
        if 'Credit_History' in df.columns:
            credit_approval = df.groupby('Credit_History')['Loan_Status'].apply(
                lambda x: (x == 'Y').mean() * 100).to_dict()
        else:
            credit_approval = {"error": "Credit_History column not found"}
        
        return {
            'approval_counts': approval_counts,
            'approval_rate': approval_rate,
            'gender_approval': gender_approval,
            'credit_approval': credit_approval
        }
    except Exception as e:
        return {"error": str(e)}

def create_visualizations(df: pd.DataFrame, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Create visualization file paths for the dataset (PNG only)
    Args:
        df: Input DataFrame
        save_dir: Directory to save the visualization files
    Returns:
        Dictionary with PNG file paths
    """
    figures = {}
    if save_dir is None:
        save_dir = "visualizations"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Loan Status Distribution (Pie Chart)
    loan_status_pie_path = os.path.join(save_dir, 'loan_status_pie.png')
    if 'Loan_Status' in df.columns:
        # Only add if file exists
        if os.path.exists(loan_status_pie_path):
            figures['loan_status_pie'] = loan_status_pie_path

    # 2. Credit History vs Loan Status
    credit_history_vs_loan_status_path = os.path.join(save_dir, 'credit_history_vs_loan_status.png')
    if os.path.exists(credit_history_vs_loan_status_path):
        figures['credit_history_vs_loan_status'] = credit_history_vs_loan_status_path

    # 3. Income by Loan Status
    income_by_loan_status_path = os.path.join(save_dir, 'income_by_loan_status.png')
    if os.path.exists(income_by_loan_status_path):
        figures['income_by_loan_status'] = income_by_loan_status_path

    # 4. Property Area vs Loan Status
    property_area_loan_status_path = os.path.join(save_dir, 'property_area_loan_status.png')
    if os.path.exists(property_area_loan_status_path):
        figures['property_area_loan_status'] = property_area_loan_status_path

    # 5. Education vs Loan Status
    education_loan_status_path = os.path.join(save_dir, 'education_loan_status.png')
    if os.path.exists(education_loan_status_path):
        figures['education_loan_status'] = education_loan_status_path

    return figures

def format_response(content: str, context: Optional[Dict] = None) -> str:
    """
    Format the response for better readability
    
    Args:
        content: The content to format
        context: Additional context information
        
    Returns:
        Formatted response string
    """
    response = f"{content}"
    
    return response