from flask import Blueprint, jsonify, request
from weather_prediction import fetch_complete_data, ml_predict
import numpy as np
import pandas as pd
import joblib

weather_bp = Blueprint('weather', __name__)

@weather_bp.route('/predict', methods=['GET'])
def predict_weather():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if lat is None or lon is None:
        return jsonify({"error": "Please provide 'lat' and 'lon' query parameters"}), 400

    start_date = request.args.get('start_date', default='2025-01-01')
    end_date = request.args.get('end_date', default='2025-01-31')

    # Fetch merged data
    merged_df = fetch_complete_data(lat, lon, start_date, end_date)
    
    # Run ML prediction and get the trained model
    model = ml_predict(merged_df)

    # Load feature names used during training
    feature_names = joblib.load("india_feature_names.pkl")

    # Prepare prediction DataFrame
    X_pred = merged_df.drop(columns=['date', 'PRECTOTCORR'], errors='ignore')

    # One-hot encode categorical columns (e.g., 'weather_condition')
    categorical_cols = X_pred.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X_pred = pd.get_dummies(X_pred, columns=categorical_cols)

    # Add missing columns from training
    for col in feature_names:
        if col not in X_pred.columns:
            X_pred[col] = 0

    # Remove extra columns not in training
    X_pred = X_pred[feature_names]

    # Ensure all values are numeric
    X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Predict
    merged_df['predicted_rainfall'] = model.predict(X_pred)
    merged_df['predicted_rainfall'] = merged_df['predicted_rainfall'].clip(lower=0)

    # Prepare final JSON result
    result = merged_df[['date', 'predicted_rainfall']].to_dict(orient='records')

    return jsonify(result)
