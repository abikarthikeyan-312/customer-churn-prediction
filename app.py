from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, send_file
import os
from functools import wraps

# Setup MySQL and Auth Extensions
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from src.models import db, User

from src.dashboard_charts import get_dashboard_data
from src.model_pipeline import trigger_retraining, get_prediction_for_customer, get_sample_customers, predict_batch
from src.shap_explainer import get_global_explanations, get_local_explanation

app = Flask(__name__)
app.secret_key = 'super_secret_churn_key_for_testing'
app.config['UPLOAD_FOLDER'] = 'data'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------
# MySQL Database Configuration
# ---------------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:password@localhost/db_name'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'error'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize DB structure manually (creates tables if they don't exist)
with app.app_context():
    db.create_all()

# ---------------------------------------------
# Authentication Routes
# ---------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            session['logged_in'] = True
            flash('Successfully logged in!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
            
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists.', 'error')
            return redirect(url_for('register'))
            
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# ---------------------------------------------
# Main Dashboard Routes
# ---------------------------------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the main dashboard page."""
    contract = request.args.get('contract', 'All')
    senior = request.args.get('senior', 'All')
    
    data = get_dashboard_data(contract_filter=contract, senior_filter=senior)
    
    if not data.get("success"):
        return render_template('dashboard.html', error=data.get("error"), active_contract=contract, active_senior=senior)
        
    return render_template('dashboard.html', 
                           metrics=data.get("metrics"), 
                           charts=data.get("charts"),
                           active_contract=contract,
                           active_senior=senior)

@app.route('/export_dashboard_csv')
@login_required
def export_dashboard_csv():
    """Exports the filtered dashboard dataset to a downloadable CSV."""
    from src.data_processing import load_data
    contract = request.args.get('contract', 'All')
    senior = request.args.get('senior', 'All')
    
    df = load_data()
    if df is not None:
        if contract and contract != 'All':
            df = df[df['Contract'] == contract]
        if senior and senior != 'All':
            senior_val = 1 if senior == 'Yes' else 0
            df = df[df['SeniorCitizen'] == senior_val]
            
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dashboard_export.csv')
        df.to_csv(temp_path, index=False)
        return send_file(temp_path, as_attachment=True, download_name="dashboard_data.csv")
        
    flash('No data available to export.', 'error')
    return redirect(url_for('dashboard'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Renders the prediction risk scoring page."""
    samples = get_sample_customers()
    
    if request.method == 'POST':
        index = request.form.get('customer_index')
        result = get_prediction_for_customer(index)
        return render_template('prediction.html', samples=samples, result=result, selected_index=int(index))
        
    return render_template('prediction.html', samples=samples, result=None)

@app.route('/predict_batch', methods=['POST'])
@login_required
def predict_batch_route():
    """Handles CSV uploads for batch prediction generation."""
    if 'batch_dataset' not in request.files:
        flash('No file provided.', 'error')
        return redirect(url_for('predict'))
        
    file = request.files['batch_dataset']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('predict'))
        
    # Save the file temporarily
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_batch.csv')
    file.save(filepath)
    
    result = predict_batch(filepath)
    if result.get("success"):
        flash(f'Successfully processed {result["count"]} customers!', 'success')
        return send_file(result["output_file"], as_attachment=True, download_name="churn_predictions.csv")
    else:
        flash(f'Batch prediction failed: {result.get("error")}', 'error')
        return redirect(url_for('predict'))


@app.route('/explain', methods=['GET', 'POST'])
@login_required
def explain():
    """Renders the SHAP Explainable AI page."""
    global_exp = get_global_explanations()
    samples = get_sample_customers()
    local_exp = None
    selected_index = None
    
    if request.method == 'POST':
        selected_index = request.form.get('customer_index')
        local_exp = get_local_explanation(selected_index)
        
    return render_template('explain.html', 
                           global_exp=global_exp, 
                           local_exp=local_exp, 
                           samples=samples,
                           selected_index=int(selected_index) if selected_index else None)

@app.route('/pipeline', methods=['GET', 'POST'])
@login_required
def pipeline():
    """Renders the ML pipeline and upload page."""
    if request.method == 'POST':
        # Handle file upload
        if 'dataset' in request.files:
            file = request.files['dataset']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_churn_data.csv')
                file.save(filepath)
                flash('Dataset uploaded successfully!', 'success')
                
        # Handle retraining
        elif 'retrain' in request.form:
            result = trigger_retraining()
            if result.get("success"):
                flash(f'Model retrained successfully! Accuracy: {result["metrics"]["accuracy"]:.2%}', 'success')
            else:
                flash(f'Error during retraining: {result.get("error")}', 'error')
                
    # Fetch historical metrics for tracking
    try:
        from src.models import ModelMetrics
        metrics_history = [m.to_dict() for m in ModelMetrics.query.order_by(ModelMetrics.training_date).all()]
    except Exception:
        metrics_history = []
        
    return render_template('pipeline.html', metrics_history=metrics_history)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
