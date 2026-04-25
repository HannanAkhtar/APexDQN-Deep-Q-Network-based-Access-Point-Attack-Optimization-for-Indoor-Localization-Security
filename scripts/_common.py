from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
from sklearn.preprocessing import StandardScaler

from search_strategies.config import MODELS_DIR
from search_strategies.core import load_wifi_split_with_augmented_val, load_noise


def load_benchmark_context(
    *,
    train_csv: str | Path = 'data/TrainingData.csv',
    val_csv: str | Path = 'data/ValidationData.csv',
    noise_file: str | Path = 'data/noise.txt',
    model_kind: Literal['nn', 'xgb'] = 'nn',
    model_path: str | Path | None = None,
):
    """Load the benchmark data, noise vector, and requested model bundle."""
    X_train, y_train, X_val, y_val = load_wifi_split_with_augmented_val(train_csv=train_csv, val_csv=val_csv)
    ap_columns = list(X_train.columns[:-4])
    noise = load_noise(noise_file, expected_length=len(X_val))

    if model_kind == 'nn':
        if model_path is None:
            model_path = MODELS_DIR / 'indoor_nn.keras'
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        scaler = StandardScaler().fit(y_train)
        predict_kwargs = {'verbose': 0}
    elif model_kind == 'xgb':
        if model_path is None:
            model_path = MODELS_DIR / 'xgb_model.pkl'
        model = joblib.load(model_path)
        scaler = None
        predict_kwargs = {}
    else:
        raise ValueError(f'Unsupported model kind: {model_kind}')

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'ap_columns': ap_columns,
        'noise': noise,
        'model': model,
        'scaler': scaler,
        'predict_kwargs': predict_kwargs,
    }
