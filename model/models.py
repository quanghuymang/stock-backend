from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stock_code = db.Column(db.String(20))
    predict_date = db.Column(db.String(20))
    actual_price = db.Column(db.Float)
    predicted_price = db.Column(db.Float)
    mse = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mae = db.Column(db.Float)
    created_at = db.Column(db.DateTime)
