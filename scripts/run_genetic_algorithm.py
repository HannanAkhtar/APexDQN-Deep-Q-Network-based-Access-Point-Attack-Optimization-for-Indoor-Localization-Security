from __future__ import annotations

import argparse

from scripts._common import load_benchmark_context
from search_strategies.algorithms import run_genetic_algorithm
from search_strategies.config import RESULTS_DIR
from search_strategies.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the genetic algorithm benchmark.')
    parser.add_argument('--train-csv', default='data/TrainingData.csv')
    parser.add_argument('--val-csv', default='data/ValidationData.csv')
    parser.add_argument('--noise-file', default='data/noise.txt')
    parser.add_argument('--model-kind', choices=['nn', 'xgb'], default='nn')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--population-size', type=int, default=30)
    parser.add_argument('--num-generations', type=int, default=30)
    parser.add_argument('--crossover-rate', type=float, default=0.8)
    parser.add_argument('--mutation-rate', type=float, default=0.15)
    parser.add_argument('--elite-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-path', default=str(RESULTS_DIR / 'genetic_algorithm_results.csv'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = load_benchmark_context(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        noise_file=args.noise_file,
        model_kind=args.model_kind,
        model_path=args.model_path,
    )
    ensure_dir(RESULTS_DIR)
    results = run_genetic_algorithm(
        ctx['model'],
        ctx['X_train'],
        ctx['X_val'],
        ctx['y_val'],
        ctx['noise'],
        ap_columns=ctx['ap_columns'],
        num_trials=args.num_trials,
        population_size=args.population_size,
        num_generations=args.num_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        seed=args.seed,
        scaler=ctx['scaler'],
        predict_kwargs=ctx['predict_kwargs'],
        output_path=args.output_path,
    )
    print(results.head())


if __name__ == '__main__':
    main()
