from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import BUILDING_ID, FLOOR, NUM_NON_AP_FEATURES, RANDOM_SEED
from .preprocessing import (
    align_validation_columns,
    augment_validation_set,
    drop_constant_and_weak,
    filter_building_floor,
    get_ap_columns,
)


def load_noise(noise_path: str | Path, expected_length: int | None = None) -> np.ndarray:
    """Load the shared perturbation vector used by the benchmark methods."""
    values = np.fromstring(Path(noise_path).read_text(), sep=' ')
    if values.size == 0:
        raise ValueError(f'No numeric noise values found in {noise_path!s}.')
    if expected_length is not None and values.size not in (1, expected_length):
        raise ValueError(
            'Noise length must be a scalar or match the validation set length. '
            f'Got {values.size}, expected 1 or {expected_length}.'
        )
    return values


def expand_noise(noise: np.ndarray | Sequence[float] | float, n_rows: int) -> tuple[float | None, np.ndarray | None]:
    """Normalize noise so downstream code can handle scalars and vectors uniformly."""
    arr = np.asarray(noise, dtype=float).reshape(-1)
    if arr.size == 1:
        return float(arr[0]), None
    if arr.size != n_rows:
        raise ValueError(f'Noise vector length {arr.size} does not match {n_rows} rows.')
    return None, arr


def load_wifi_split(
    train_csv: str | Path = 'data/TrainingData.csv',
    val_csv: str | Path = 'data/ValidationData.csv',
    *,
    building_id: int = BUILDING_ID,
    floor: int = FLOOR,
    weak_signal_threshold: float = -90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and filter the benchmark split used across the repository."""
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    X_train, y_train, X_val, y_val = filter_building_floor(df_train, df_val)
    X_train = drop_constant_and_weak(X_train, weak_signal_threshold=weak_signal_threshold)
    X_val = align_validation_columns(X_train, X_val)
    return X_train, y_train, X_val, y_val


def load_wifi_split_with_augmented_val(
    train_csv: str | Path = 'data/TrainingData.csv',
    val_csv: str | Path = 'data/ValidationData.csv',
    *,
    building_id: int = BUILDING_ID,
    floor: int = FLOOR,
    weak_signal_threshold: float = -90,
    val_aug_size: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the split and augment the validation set with train samples."""
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    X_train, y_train, X_val, y_val = filter_building_floor(df_train, df_val)
    X_train = drop_constant_and_weak(X_train, weak_signal_threshold=weak_signal_threshold)
    X_val = align_validation_columns(X_train, X_val)
    X_train, y_train, X_val, y_val = augment_validation_set(X_train, y_train, X_val, y_val, size=val_aug_size)
    return X_train, y_train, X_val, y_val


def apply_attack_noise(
    X: pd.DataFrame,
    attack_aps: Sequence[str],
    noise: np.ndarray | Sequence[float] | float,
) -> pd.DataFrame:
    """Return a copy of X with shared noise added to the selected AP columns."""
    attacked = X.copy()
    scalar_noise, vector_noise = expand_noise(noise, len(attacked))
    for ap in attack_aps:
        if ap not in attacked.columns:
            continue
        if scalar_noise is not None:
            attacked[ap] += scalar_noise
        else:
            attacked[ap] += vector_noise
    return attacked


def predict_values(model, X: pd.DataFrame, *, scaler=None, predict_kwargs: dict | None = None) -> np.ndarray:
    """Run model inference and optionally inverse-transform the output."""
    predict_kwargs = predict_kwargs or {}
    predictions = model.predict(X, **predict_kwargs)
    predictions = np.asarray(predictions)
    if scaler is not None:
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
    return np.asarray(predictions)


def evaluate_attack_mse(
    model,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    attack_aps: Sequence[str],
    noise: np.ndarray | Sequence[float] | float,
    *,
    scaler=None,
    predict_kwargs: dict | None = None,
) -> float:
    """Compute the attacked-model MSE for a specific AP subset."""
    attacked = apply_attack_noise(X_val, attack_aps, noise)
    predictions = predict_values(model, attacked, scaler=scaler, predict_kwargs=predict_kwargs)
    target = np.asarray(y_val)
    return float(np.mean((predictions - target) ** 2))


def format_combo(combo: Sequence[str] | None) -> str:
    if not combo:
        return ''
    return ','.join(combo)
