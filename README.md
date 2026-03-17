# Customer Churn Prediction

A Flask web application for customer churn prediction, including data upload, model training pipeline, dashboard visualizations, and SHAP explainability.

## Project Structure
- `app.py` - Flask app routes and endpoints
- `src/` - Data processing, model pipeline, and explainability modules
- `templates/` - HTML templates
- `static/` - CSS and static assets
- `data/` - Input and output CSV files
- `models/` - Saved model files

## Setup
1. Create a Python venv:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open browser at `http://127.0.0.1:5000`

## Usage
- Register/log in
- Upload customer churn dataset
- Train model and evaluate accuracy
- View dashboard and SHAP explanations

## Author
- GitHub: [abikarthikeyan-312](https://github.com/abikarthikeyan-312)
