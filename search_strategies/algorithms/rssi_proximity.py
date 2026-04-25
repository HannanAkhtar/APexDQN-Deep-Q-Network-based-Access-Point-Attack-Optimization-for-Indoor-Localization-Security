from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core import evaluate_attack_mse, format_combo
from ..preprocessing import get_ap_columns


def compute_rssi_probabilities(X_train: pd.DataFrame, ap_columns: Sequence[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Compute RSSI-weighted AP sampling probabilities from the training set."""
    ap_columns = list(ap_columns or get_ap_columns(X_train))
    ap_mean_rssi: dict[str, float] = {}
    for ap in ap_columns:
        detected = X_train[ap][X_train[ap] != 100]
        ap_mean_rssi[ap] = float(detected.mean()) if len(detected) > 0 else -999.0

    min_rssi = min(ap_mean_rssi.values())
    weights = np.array([ap_mean_rssi[ap] - min_rssi + 1 for ap in ap_columns], dtype=float)
    probabilities = weights / weights.sum()
    return ap_columns, probabilities


def run_rssi_proximity(
    model,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    noise,
    *,
    ap_columns: Sequence[str] | None = None,
    num_trials: int = 500,
    seed: int = 42,
    scaler=None,
    predict_kwargs: dict | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Sample AP subsets proportionally to RSSI strength and record each trial."""
    rng = np.random.default_rng(seed)
    ap_columns, probabilities = compute_rssi_probabilities(X_train, ap_columns=ap_columns)
    rows: list[dict] = []

    for k in range(1, len(ap_columns) + 1):
        for trial in range(1, num_trials + 1):
            start_time = time.time()
            attack_aps = list(rng.choice(ap_columns, size=k, replace=False, p=probabilities))
            mse = evaluate_attack_mse(
                model,
                X_val,
                y_val,
                attack_aps,
                noise,
                scaler=scaler,
                predict_kwargs=predict_kwargs,
            )
            rows.append({
                'Num_Attacked_APs': k,
                'Trial': trial,
                'Best_Combination': format_combo(attack_aps),
                'Best_MSE': mse,
                'Time_Taken_sec': time.time() - start_time,
            })

    results = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
    return results
