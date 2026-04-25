from __future__ import annotations

"""Driver to run DQN-based search for k=1..K using a frozen baseline."""

import argparse
import joblib
from pathlib import Path
import time

from search_strategies.data import load_raw
from search_strategies.preprocessing import filter_building_floor, drop_constant_and_weak, align_validation_columns, stitch_500_from_train, ap_only_columns
from search_strategies.algorithms.noise import read_noise
from search_strategies.algorithms.rl.train_dqn import train_dqn_for_k
from search_strategies.utils.io import ensure_dir
from search_strategies.config import MODELS_DIR, RESULTS_DIR, NOISE_TXT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['nn', 'xgb'], default='xgb')
    parser.add_argument('--kmin', type=int, default=1)
    parser.add_argument('--kmax', type=int, default=21)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--xgb_pkl', type=str, default=None)
    args = parser.parse_args()

    ensure_dir(RESULTS_DIR)
    df_train, df_val = load_raw()
    Xtr, ytr, Xva, yva = filter_building_floor(df_train, df_val)
    Xtr = drop_constant_and_weak(Xtr)
    Xva = align_validation_columns(Xtr, Xva)
    Xtr, ytr, Xva, yva = stitch_500_from_train(Xtr, ytr, Xva.values, yva.values)

    if args.backend == 'xgb':
        pkl = Path(args.xgb_pkl) if args.xgb_pkl else (MODELS_DIR / 'xgb_model.pkl')
        if pkl.exists():
            xgb_model = joblib.load(pkl)
        else:
            from search_strategies.models.xgb_model import fit_and_eval_xgb
            xgb_model, _ = fit_and_eval_xgb(Xtr, ytr, Xva, yva, model_path=None)
        baseline = xgb_model
    else:
        nn_path = MODELS_DIR / 'indoor_nn.keras'
        scaler_path = MODELS_DIR / 'scaler_output.joblib'
        if not nn_path.exists() or not Path(scaler_path).exists():
            raise RuntimeError('Please run scripts.train_baselines and save NN + scaler before running DQN-based search with backend=nn')
        import tensorflow as tf
        nn_model = tf.keras.models.load_model(nn_path)
        scaler_out = joblib.load(scaler_path)
        baseline = (nn_model, scaler_out)

    noise_is_scalar, noise_value = read_noise(NOISE_TXT, n_samples=Xva.shape[0])
    ap_cols = ap_only_columns(Xtr)

    out_path = RESULTS_DIR / 'dqn_results.txt'
    with open(out_path, 'w') as f:
        f.write('Num_Attacked_APs,Best_Combination,MSE,Time_Taken_sec\n')

    for k in range(args.kmin, args.kmax + 1):
        t0 = time.time()
        best_combo, best_mse, _ = train_dqn_for_k(
            baseline_backend=args.backend,
            baseline_model=baseline,
            X_val=Xva,
            y_val=yva,
            ap_columns=ap_cols,
            k=k,
            noise_is_scalar=noise_is_scalar,
            noise_value=noise_value,
            num_episodes=args.episodes,
        )
        dt = time.time() - t0
        combo_str = ','.join(best_combo) if best_combo else ''
        with open(out_path, 'a') as f:
            f.write(f'{k},"{combo_str}",{best_mse:.6f},{dt:.2f}\n')
        print(f'k={k} best {combo_str} mse={best_mse:.6f} time={dt:.2f}s')

    print(f'Wrote DQN-based search results to {out_path}')


if __name__ == '__main__':
    main()
