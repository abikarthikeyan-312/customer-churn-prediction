import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path="data/uploaded_churn_data.csv"):
    """Loads raw data, returns None if not found."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None

def preprocess_data(df, require_churn=True):
    """
    Cleans and preprocesses the Telco Churn DataFrame.
    Returns: X (features), y (target), label_encoders (for inverse transform if needed), 
             and the original df with basic cleaning.
    """
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()
    
    # 1. Drop irrelevant columns
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)
        
    # 2. Handle TotalCharges (sometimes it's string with spaces for new customers)
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # Fill NaN with 0 or median (0 makes sense for 0 tenure)
        data['TotalCharges'] = data['TotalCharges'].fillna(0)
        
    # 3. Separate features and target
    if require_churn and 'Churn' not in data.columns:
        raise ValueError("The dataset must contain a 'Churn' column (Yes/No target) to train the model. Found columns: " + ", ".join(data.columns.tolist()))
        
    if 'Churn' in data.columns:
        y = data['Churn'].map({'Yes': 1, 'No': 0})
        X = data.drop('Churn', axis=1)
    else:
        y = None
        X = data
    
    # 4. Encoding categorical variables
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # For tree based models, label encoding is often sufficient, 
    # but we'll use pd.get_dummies for better SHAP interpretability
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Ensure boolean columns are integers to avoid issues with some xgb versions
    for col in X_encoded.columns:
        if X_encoded[col].dtype == bool:
            X_encoded[col] = X_encoded[col].astype(int)
            
    return X_encoded, y, label_encoders, data

def get_base_metrics(df: pd.DataFrame):
    """Calculate basic KPIs for the dashboard"""
    if df is None or len(df) == 0:
        return {}
        
    total_customers = len(df)
    churn_rate = float((df['Churn'] == 'Yes').mean()) * 100 if 'Churn' in df.columns else 0.0
    avg_lifetime = float(df['tenure'].mean()) if 'tenure' in df.columns else 0.0
    avg_mrr = float(df['MonthlyCharges'].mean()) if 'MonthlyCharges' in df.columns else 0.0
    
    return {
        "total_customers": total_customers,
        "churn_rate": churn_rate,
        "avg_lifetime": avg_lifetime,
        "avg_mrr": avg_mrr
    }
