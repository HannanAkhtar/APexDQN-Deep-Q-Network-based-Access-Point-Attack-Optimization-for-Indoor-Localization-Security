from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core import evaluate_attack_mse, format_combo
from ..preprocessing import get_ap_columns


def create_individual(all_aps: Sequence[str], k: int, rng: np.random.Generator | None = None) -> list[str]:
    rng = rng or np.random.default_rng()
    return list(rng.choice(list(all_aps), size=k, replace=False))


def crossover(parent1: Sequence[str], parent2: Sequence[str], all_aps: Sequence[str], k: int, rng: np.random.Generator | None = None) -> list[str]:
    rng = rng or np.random.default_rng()
    combined = list(dict.fromkeys(list(parent1) + list(parent2)))
    if len(combined) >= k:
        return list(rng.choice(combined, size=k, replace=False))
    remaining = [ap for ap in all_aps if ap not in combined]
    extra = list(rng.choice(remaining, size=k - len(combined), replace=False))
    return combined + extra


def mutate(individual: Sequence[str], all_aps: Sequence[str], k: int, mutation_rate: float, rng: np.random.Generator | None = None) -> list[str]:
    rng = rng or np.random.default_rng()
    mutated = list(individual)
    for idx in range(k):
        if rng.random() < mutation_rate:
            current_set = set(mutated)
            remaining = [ap for ap in all_aps if ap not in current_set]
            if remaining:
                mutated[idx] = rng.choice(remaining)
    return mutated


def tournament_selection(population: Sequence[Sequence[str]], fitness_scores: Sequence[float], tournament_size: int = 3, rng: np.random.Generator | None = None) -> list[str]:
    rng = rng or np.random.default_rng()
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
    return list(population[best_idx])


def run_genetic_algorithm(
    model,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    noise,
    *,
    ap_columns: Sequence[str] | None = None,
    num_trials: int = 10,
    population_size: int = 30,
    num_generations: int = 30,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.15,
    elite_size: int = 2,
    seed: int = 42,
    scaler=None,
    predict_kwargs: dict | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the genetic algorithm benchmark and return one summary row per k."""
    rng = np.random.default_rng(seed)
    ap_columns = list(ap_columns or get_ap_columns(X_train))
    rows: list[dict] = []

    for k in range(1, len(ap_columns) + 1):
        start_time = time.time()
        trial_best_mses: list[float] = []
        overall_best_mse = -float('inf')
        overall_best_combo: list[str] | None = None

        for _ in range(num_trials):
            population = [create_individual(ap_columns, k, rng=rng) for _ in range(population_size)]
            fitness_scores = [
                evaluate_attack_mse(model, X_val, y_val, ind, noise, scaler=scaler, predict_kwargs=predict_kwargs)
                for ind in population
            ]

            best_idx = int(np.argmax(fitness_scores))
            best_aps = list(population[best_idx])
            best_mse = float(fitness_scores[best_idx])

            for _ in range(num_generations):
                sorted_indices = np.argsort(fitness_scores)[::-1]
                new_population = [list(population[sorted_indices[i]]) for i in range(min(elite_size, len(population)))]

                while len(new_population) < population_size:
                    parent1 = tournament_selection(population, fitness_scores, rng=rng)
                    parent2 = tournament_selection(population, fitness_scores, rng=rng)
                    if rng.random() < crossover_rate:
                        child = crossover(parent1, parent2, ap_columns, k, rng=rng)
                    else:
                        child = list(parent1)
                    child = mutate(child, ap_columns, k, mutation_rate, rng=rng)
                    new_population.append(child)

                population = new_population
                fitness_scores = [
                    evaluate_attack_mse(model, X_val, y_val, ind, noise, scaler=scaler, predict_kwargs=predict_kwargs)
                    for ind in population
                ]

                current_best_idx = int(np.argmax(fitness_scores))
                current_best_mse = float(fitness_scores[current_best_idx])
                if current_best_mse > best_mse:
                    best_mse = current_best_mse
                    best_aps = list(population[current_best_idx])

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
