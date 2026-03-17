import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import io
import base64
from src.data_processing import load_data, preprocess_data

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_churn_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")

def train_model(X, y):
    """Trains an XGBoost model and saves it."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }
    
    # Save model and features list
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)
    
    # Log metrics to MySQL Database (Feature 3)
    try:
        from flask import current_app
        if current_app:
            from src.models import db, ModelMetrics
            import json
            metric_record = ModelMetrics(
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1"],
                configuration=json.dumps({
                    "n_estimators": 100, 
                    "learning_rate": 0.1, 
                    "max_depth": 5
                })
            )
            db.session.add(metric_record)
            db.session.commit()
    except Exception:
        pass # Ignore if not inside a Flask request context
        
    return model, metrics

def load_model():
    """Returns model and expected features."""
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)
    return None, None

def trigger_retraining():
    """Triggers retraining logic, returns metrics or error."""
    df = load_data()
    if df is None:
        return {"success": False, "error": "No data found."}
        
    try:
        X, y, _, _ = preprocess_data(df)
        if X is None:
            return {"success": False, "error": "Invalid data structure."}
            
        model, metrics = train_model(X, y)
        return {"success": True, "metrics": metrics, "size": len(df)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"An error occurred during retraining: {str(e)}"}

def _parse_customer_index(index, max_index):
    """Validates and parses the customer index from input."""
    if index is None or str(index).strip() == "":
        raise ValueError("Customer index is required.")
    try:
        idx = int(index)
    except (TypeError, ValueError):
        raise ValueError("Customer index must be an integer.")
    if idx < 0 or idx >= max_index:
        raise IndexError(f"Customer index must be between 0 and {max_index - 1}.")
    return idx


def get_prediction_for_customer(index):
    """Predict risk for a customer at a given index"""
    model, features = load_model()
    if model is None or features is None:
        return {"success": False, "error": "Model not trained or model files missing."}

    df = load_data()
    if df is None:
        return {"success": False, "error": "Data not found"}

    try:
        X, y, _, _ = preprocess_data(df)
        cust_idx = _parse_customer_index(index, len(X))

        cust_X = X.iloc[[cust_idx]].copy()
        cust_X = cust_X.reindex(columns=features, fill_value=0)

        prob = float(model.predict_proba(cust_X)[0][1] * 100)

        orig_data = df.iloc[[cust_idx]].to_dict('records')[0]
        # Convert any numpy types to native Python types for JSON serialization
        import numpy as np
        for k, v in orig_data.items():
            if isinstance(v, (np.integer, np.floating)):
                orig_data[k] = v.item()

        return {
            "success": True,
            "probability": round(prob, 2),
            "risk_label": "High Risk" if prob > 50 else "Low Risk",
            "customer_data": orig_data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def predict_batch(file_path):
    """Predicts churn for a batch of customers from a CSV file."""
    model, features = load_model()
    if model is None or features is None:
        return {"success": False, "error": "Model not trained."}
        
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {str(e)}"}
        
    try:
        X, _, _, orig_df = preprocess_data(df, require_churn=False)
        
        # Align features
        X = X.reindex(columns=features, fill_value=0)
        
        # Predict Proba
        probs = model.predict_proba(X)[:, 1] * 100
        
        # Append predictions to the original data (not the one-hot encoded one)
        orig_df['Churn_Probability'] = probs.round(2)
        orig_df['Risk_Label'] = ['High Risk' if p > 50 else 'Low Risk' for p in probs]
        
        # Save to output file
        output_path = os.path.join("data", "batch_predictions_result.csv")
        orig_df.to_csv(output_path, index=False)
        
        return {"success": True, "output_file": output_path, "count": len(orig_df)}
        
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}"}

def get_sample_customers():
    df = load_data()
    if df is None:
        return []
    # Return basic info of first 50 customers to populate dropdowns
    samples = df.head(50).to_dict('records')
    for i, s in enumerate(samples):
        s['index'] = i
    return samples
