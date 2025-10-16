import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

def fit_and_eval_xgb(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, model_path: Path | None = None):
    """Trains or loads an XGBoost model and evaluates its performance.

    If a path to a pre-trained model is provided, it loads the model;
    otherwise, it trains a new model from scratch.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.DataFrame): Training target data.
        X_val (pd.DataFrame): Validation feature data.
        y_val (pd.DataFrame): Validation target data.
        model_path (Path | None): Optional path to load a pre-trained .pkl model.

    Returns:
        tuple: A tuple containing:
            - model (XGBRegressor): The trained or loaded XGBoost model.
            - results (dict): A dictionary with MSE and MAE performance metrics.
    """
    if model_path and model_path.exists():
        print(f"Loading pre-trained XGBoost model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Training new XGBoost model.")
        model = XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)

    # Evaluate the model on the validation set.
    preds = model.predict(X_val)
    mse = float(mean_squared_error(y_val, preds))
    mae = float(mean_absolute_error(y_val, preds))
    print(f"XGBoost Validation => MSE: {mse:.6f} | MAE: {mae:.6f}")

    return model, dict(mse=mse, mae=mae)

