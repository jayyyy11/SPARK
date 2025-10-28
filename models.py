from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Initialize the database extension
db = SQLAlchemy()

class User(UserMixin, db.Model):
    """
    Stores user account information.
    - UserMixin adds the properties Flask-Login expects (like is_authenticated).
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    # This 'relationship' links the User to all their 'Analysis' records.
    # 'backref' adds an 'user' attribute to the Analysis model
    # 'lazy=True' means SQLAlchemy will load the analyses as needed.
    analyses = db.relationship('Analysis', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Analysis(db.Model):
    """
    Stores one record of a voice analysis, linked to a user.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # We store the *paths* to the files, not the files themselves
    audio_filename = db.Column(db.String(300), nullable=False)
    report_filename = db.Column(db.String(300), nullable=False)
    
    # The results of the analysis
    prediction_result = db.Column(db.String(50), nullable=False) # "Positive" or "Negative"
    prediction_proba = db.Column(db.Float, nullable=False)      # The % confidence
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Analysis {self.id} - {self.prediction_result}>'