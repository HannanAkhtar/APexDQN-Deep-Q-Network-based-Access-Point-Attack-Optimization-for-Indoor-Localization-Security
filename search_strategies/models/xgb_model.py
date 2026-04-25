from __future__ import annotations

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path


def fit_and_eval_xgb(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, model_path: Path | None = None):
    """Trains or loads an XGBoost model and evaluates its performance."""
    if model_path and model_path.exists():
        print(f'Loading pre-trained XGBoost model from {model_path}')
        model = joblib.load(model_path)
    else:
        print('Training new XGBoost model.')
        model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mse = float(mean_squared_error(y_val, preds))
    mae = float(mean_absolute_error(y_val, preds))
    print(f'XGBoost Validation => MSE: {mse:.6f} | MAE: {mae:.6f}')
    return model, dict(mse=mse, mae=mae)
