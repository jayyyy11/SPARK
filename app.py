import os
import joblib
import pdfkit  # To create the PDF report
from flask import (
    Flask, render_template, request, redirect, url_for, flash, 
    send_from_directory, abort
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from models import db, User, Analysis  # Import from our new models.py
from helper import extract_voice_features # Import our audio helper
import numpy as np
from datetime import datetime  # <-- FIX 1: IMPORT DATETIME

# --- FIX 2: HARD-CODE THE PATH TO WKHTMLTOPDF ---
# This is the "guaranteed" fix for the PDF generation error.
# Double-check this path matches where you installed the .exe
path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
# --------------------------------------------------


# --- 1. App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key-please-change-this' # IMPORTANT: Change this!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.instance_path, 'app.db')
app.config['UPLOAD_FOLDER'] = os.path.join('uploads')
app.config['REPORT_FOLDER'] = os.path.join('reports')

# Ensure instance, upload, and report folders exist
try:
    os.makedirs(app.instance_path)
except OSError:
    pass
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# --- 2. Database & Login Setup ---
db.init_app(app)
migrate = Migrate(app, db) # This handles database migrations
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Page to redirect to if not logged in
login_manager.login_message_category = 'info' # Bootstrap category for flash message

@login_manager.user_loader
def load_user(user_id):
    # Flask-Login uses this to get the user object for the session
    return User.query.get(int(user_id))

# --- 3. Load ML Model & Scaler ---
# We load the "brain" once when the app starts
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("ML model and scaler loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: 'best_model.pkl' or 'scaler.pkl' not found.")
    print("Please run build_model.py first.")
    model, scaler = None, None
except Exception as e:
    print(f"Error loading model files: {e}")
    model, scaler = None, None

# --- 4. Authentication Routes (Login, Register, Logout) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard')) # Go to dashboard if already logged in
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first() # Find the user
        
        if user and user.check_password(password):
            login_user(user) # Log them in
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html') # Show login page on GET request

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'warning')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html') # Show registration page

@app.route('/logout')
@login_required # This route requires the user to be logged in
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- 5. Core Application Routes (Dashboard & Analysis) ---
@app.route('/')
def index():
    # Main page: send logged-in users to dashboard, others to login
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get all past analyses for *this* user, most recent first
    user_analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).all()
    return render_template('dashboard.html', analyses=user_analyses)

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'audio_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dashboard'))
    
    file = request.files['audio_file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard'))
        
    if not (file and model and scaler):
        flash('Model is not loaded. Cannot perform analysis.', 'danger')
        return redirect(url_for('dashboard'))
        
    filename = secure_filename(file.filename)
    
    # --- 1. Save Audio File ---
    # Create a user-specific folder (e.g., /uploads/user_1/)
    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{current_user.id}")
    os.makedirs(user_upload_dir, exist_ok=True)
    audio_path = os.path.join(user_upload_dir, filename)
    file.save(audio_path)
    
    # --- 2. Run Prediction Pipeline ---
    # Use the SAME helper.py function from training
    features = extract_voice_features(audio_path)
    if features is None:
        flash('Could not extract audio features. Please try a different file.', 'danger')
        return redirect(url_for('dashboard'))
        
    # Use the SAME scaler.pkl from training
    scaled_features = scaler.transform(features)
    
    # Use the SAME best_model.pkl from training
    prediction = model.predict(scaled_features)[0]       # Result (0 or 1)
    probability = model.predict_proba(scaled_features)[0] # Probabilities [P(0), P(1)]
    
    proba_parkinsons = probability[1] # Get the probability of class '1' (Parkinson's)
    result_text = "Positive" if prediction == 1 else "Negative"
    
    # --- 3. Generate PDF Report ---
    # Render an HTML template with the results
    report_html = render_template(
        'report_template.html', 
        user=current_user,
        result=result_text,
        probability=f"{proba_parkinsons*100:.2f}",
        filename=filename,
        analysis_date=datetime.now().strftime('%Y-%m-%d %H:%M') # <-- Uses 'datetime' import
    )
    
    # Create user-specific report folder (e.g., /reports/user_1/)
    user_report_dir = os.path.join(app.config['REPORT_FOLDER'], f"user_{current_user.id}")
    os.makedirs(user_report_dir, exist_ok=True)
    
    report_filename = f"report_{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf" # <-- Uses 'datetime' import
    report_path = os.path.join(user_report_dir, report_filename)
    
    try:
        # --- FIX 2 (Continued): Use the 'configuration=config' ---
        pdfkit.from_string(report_html, report_path, configuration=config)
        
    except IOError:
        flash('Could not generate PDF report. Is wkhtmltopdf installed and configured correctly?', 'danger')
        return redirect(url_for('dashboard'))
    
    # --- 4. Save to Database ---
    # Store the paths relative to the upload/report folders
    audio_file_path_db = os.path.join(f"user_{current_user.id}", filename)
    report_file_path_db = os.path.join(f"user_{current_user.id}", report_filename)

    new_analysis = Analysis(
        user_id=current_user.id,
        audio_filename=audio_file_path_db,
        report_filename=report_file_path_db,
        prediction_result=result_text,
        prediction_proba=proba_parkinsons
    )
    db.session.add(new_analysis)
    db.session.commit()
    
    flash('Analysis complete! Your report is ready.', 'success')
    return redirect(url_for('dashboard'))

# --- 6. Download Routes (for audio and reports) ---
@app.route('/download_report/<int:analysis_id>')
@login_required
def download_report(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    # Security: Ensure user can only download their own reports
    if analysis.user_id != current_user.id:
        abort(403) # Forbidden
        
    directory = app.config['REPORT_FOLDER']
    return send_from_directory(directory, analysis.report_filename, as_attachment=True)
    
@app.route('/download_audio/<int:analysis_id>')
@login_required
def download_audio(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        abort(403) # Forbidden
        
    directory = app.config['UPLOAD_FOLDER']
    return send_from_directory(directory, analysis.audio_filename, as_attachment=True)


# --- 7. Run the App ---
if __name__ == '__main__':
    with app.app_context():
        # Create database tables if they don't exist
        db.create_all()
    app.run(debug=True)