from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ModelMetrics(db.Model):
    __tablename__ = 'model_metrics'
    id = db.Column(db.Integer, primary_key=True)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    configuration = db.Column(db.Text, nullable=True) # Stores JSON str of hyperparams

    def to_dict(self):
        return {
            "id": self.id,
            "training_date": self.training_date.strftime('%Y-%m-%d %H:%M:%S'),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "configuration": json.loads(self.configuration) if self.configuration else {}
        }
