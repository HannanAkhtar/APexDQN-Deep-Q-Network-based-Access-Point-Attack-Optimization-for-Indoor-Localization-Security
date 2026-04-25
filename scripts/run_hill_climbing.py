from __future__ import annotations

import argparse

from scripts._common import load_benchmark_context
from search_strategies.algorithms import run_hill_climbing
from search_strategies.config import RESULTS_DIR
from search_strategies.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the hill-climbing benchmark.')
    parser.add_argument('--train-csv', default='data/TrainingData.csv')
    parser.add_argument('--val-csv', default='data/ValidationData.csv')
    parser.add_argument('--noise-file', default='data/noise.txt')
    parser.add_argument('--model-kind', choices=['nn', 'xgb'], default='nn')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--num-trials', type=int, default=1)
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-path', default=str(RESULTS_DIR / 'hill_climbing_results.csv'))
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
    results = run_hill_climbing(
        ctx['model'],
        ctx['X_train'],
        ctx['X_val'],
        ctx['y_val'],
        ctx['noise'],
        ap_columns=ctx['ap_columns'],
        num_trials=args.num_trials,
        max_iterations=args.max_iterations,
        seed=args.seed,
        scaler=ctx['scaler'],
        predict_kwargs=ctx['predict_kwargs'],
        output_path=args.output_path,
    )
    print(results.head())


if __name__ == '__main__':
    main()
