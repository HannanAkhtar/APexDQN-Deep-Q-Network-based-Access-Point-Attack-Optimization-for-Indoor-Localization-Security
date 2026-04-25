from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core import evaluate_attack_mse, format_combo
from ..preprocessing import get_ap_columns


def get_neighbor(current_aps: Sequence[str], all_aps: Sequence[str], k: int, rng: np.random.Generator | None = None) -> list[str]:
    """Generate a neighbor by swapping one AP in the current set with one outside it."""
    rng = rng or np.random.default_rng()
    current_aps = list(current_aps)
    current_set = set(current_aps)
    remaining = [ap for ap in all_aps if ap not in current_set]
    if not remaining:
        return current_aps

    ap_to_remove = rng.choice(current_aps)
    ap_to_add = rng.choice(remaining)
    return [ap if ap != ap_to_remove else ap_to_add for ap in current_aps]


def run_hill_climbing(
    model,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    noise,
    *,
    ap_columns: Sequence[str] | None = None,
    num_trials: int = 1,
    max_iterations: int = 100,
    seed: int = 42,
    scaler=None,
    predict_kwargs: dict | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the hill-climbing benchmark and return one summary row per k."""
    np.random.seed(seed)
    ap_columns = list(ap_columns or get_ap_columns(X_train))
    rows: list[dict] = []

    for k in range(1, len(ap_columns) + 1):
        start_time = time.time()
        trial_best_mses: list[float] = []
        overall_best_mse = -float('inf')
        overall_best_combo: list[str] | None = None

        for _ in range(num_trials):
            current_aps = list(np.random.choice(ap_columns, size=k, replace=False))
            current_mse = evaluate_attack_mse(
                model,
                X_val,
                y_val,
                current_aps,
                noise,
                scaler=scaler,
                predict_kwargs=predict_kwargs,
            )
            best_aps = list(current_aps)
            best_mse = current_mse

            for _ in range(max_iterations):
                neighbor_aps = get_neighbor(current_aps, ap_columns, k, rng=np.random.default_rng())
                neighbor_mse = evaluate_attack_mse(
                    model,
                    X_val,
                    y_val,
                    neighbor_aps,
                    noise,
                    scaler=scaler,
                    predict_kwargs=predict_kwargs,
                )
                if neighbor_mse > current_mse:
                    current_aps = neighbor_aps
                    current_mse = neighbor_mse
                    if current_mse > best_mse:
                        best_mse = current_mse
                        best_aps = list(current_aps)

            trial_best_mses.append(best_mse)
            if best_mse > overall_best_mse:
                overall_best_mse = best_mse
                overall_best_combo = list(best_aps)

        rows.append({
            'Num_Attacked_APs': k,
            'Best_Combination': format_combo(overall_best_combo),
            'Best_MSE': overall_best_mse,
            'Avg_MSE_over_trials': float(np.mean(trial_best_mses)),
            'Time_Taken_sec': time.time() - start_time,
        })

    results = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
    return results
