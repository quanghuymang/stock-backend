from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from model.lstm_model import predict_with_lstm
from model.models import db, PredictionHistory
import csv
from io import BytesIO
from flask import send_file
import pandas as pd
from flask import Response
import os  # thêm dòng này ở đầu file nếu chưa có

app = Flask(__name__)

CORS(app)

# Cấu hình kết nối MySQL (bạn cần sửa user/pass/dbname cho phù hợp)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/stock_predictor'

#db online 
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://sql12770258:LsK7Hvu3nm@sql12.freesqldatabase.com/sql12770258'

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = predict_with_lstm(data)

    prediction = PredictionHistory(
        stock_code=data.get("stockCode"),
        predict_date=result["date"],
        actual_price=result["actualPrice"],
        predicted_price=result["predictedPrice"],
        mse=result["metrics"]["mse"],
        rmse=result["metrics"]["rmse"],
        mae=result["metrics"]["mae"],
        created_at=datetime.now()
    )
    db.session.add(prediction)
    db.session.commit()

    return jsonify(result)

@app.route("/history", methods=["GET"])
def get_history():
    predictions = PredictionHistory.query.all()
    result = [{
        "stock_code": pred.stock_code,
        "predict_date": pred.predict_date,
        "actual_price": pred.actual_price,
        "predicted_price": pred.predicted_price,
        "mse": pred.mse,
        "rmse": pred.rmse,
        "mae": pred.mae,
        "created_at": pred.created_at.strftime("%Y-%m-%d %H:%M:%S")
    } for pred in predictions]
    return jsonify(result)

from io import BytesIO
from flask import send_file

@app.route("/export", methods=["GET"])
def export_history():
    predictions = PredictionHistory.query.all()
    data = [{
        "stock_code": pred.stock_code,
        "predict_date": pred.predict_date,
        "actual_price": pred.actual_price,
        "predicted_price": pred.predicted_price,
        "mse": pred.mse,
        "rmse": pred.rmse,
        "mae": pred.mae,
        "created_at": pred.created_at.strftime("%Y-%m-%d %H:%M:%S")
    } for pred in predictions]

    df = pd.DataFrame(data)

    # Sử dụng BytesIO để tạo buffer in-memory CSV
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)  # Quay lại đầu buffer

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv"
    )

    
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
