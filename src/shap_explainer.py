import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from src.model_pipeline import load_model
from src.data_processing import preprocess_data, load_data

def get_base64_image(fig):
    """Utility to convert a matplotlib figure to a base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig) # free memory
    return f"data:image/png;base64,{img_b64}"

def get_global_explanations():
    """Generates the global feature importance SHAP plot as a base64 string."""
    model, features = load_model()
    if not model:
        return {"success": False, "error": "No model found."}
        
    df = load_data()
    if df is None:
        return {"success": False, "error": "No data found."}
        
    X, y, _, _ = preprocess_data(df)
    if X is None:
        return {"success": False, "error": "Data processing error."}
    
    # Global Importance
    X_sample = shap.sample(X, 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("dark_background")
    shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
    
    # Tweak styling for web
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    plt.tight_layout()
    
    b64_img = get_base64_image(fig)
    return {"success": True, "global_plot": b64_img}

def get_local_explanation(index):
    """Generates the local waterfall SHAP plot for a specific customer."""
    model = load_model()[0]
    df = load_data()

    if model is None or df is None:
        return {"success": False, "error": "Environment missing."}

    X, y, _, _ = preprocess_data(df)

    try:
        if index is None or str(index).strip() == "":
            raise ValueError("Customer index is required for local explanation.")
        cust_idx = int(index)
        if cust_idx < 0 or cust_idx >= len(X):
            raise IndexError(f"Customer index must be between 0 and {len(X)-1}.")

        cust_X = X.iloc[[cust_idx]]
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for the single customer row
        cust_shap_values = explainer(cust_X)
        
        # Check SHAP versions / output shapes depending on model objective
        if len(cust_shap_values.values.shape) == 3:
            # Multi-class output: shape (1, num_features, classes) -> Get class 1 (Churn)
            vals = cust_shap_values[0, :, 1].values
            base_val = cust_shap_values[0, :, 1].base_values
        elif len(cust_shap_values.values.shape) == 2:
            # Binary output: shape (1, num_features)
            vals = cust_shap_values[0].values
            base_val = cust_shap_values[0].base_values
        else:
            vals = cust_shap_values.values[0]
            base_val = cust_shap_values.base_values
            
        # Ensure base_val is scalar
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]

        # Reconstruct exactly 1D explanation for waterfall plotting
        explanation = shap.Explanation(
            values=vals,
            base_values=base_val,
            data=cust_X.iloc[0].values,
            feature_names=cust_X.columns.tolist()
        )

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.patch.set_facecolor('#0f172a')

        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()

        b64_img = get_base64_image(fig2)
        return {"success": True, "local_plot": b64_img}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
