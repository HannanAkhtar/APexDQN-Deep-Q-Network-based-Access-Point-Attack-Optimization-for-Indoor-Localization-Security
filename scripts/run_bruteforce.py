"""
Run brute-force attack enumeration for k = 1..21 (or any range).
This unified script supports both NN and XGB backends.
Usage:
    python -m scripts.run_bruteforce --backend nn
    python -m scripts.run_bruteforce --backend xgb
"""
import argparse
from pathlib import Path
import joblib
import time

from apexdqn.data import load_raw
from apexdqn.preprocessing import filter_building_floor, drop_constant_and_weak, align_validation_columns, stitch_500_from_train, ap_only_columns
from apexdqn.attacks.noise import read_noise
from apexdqn.attacks.brute_force import brute_force_evaluate
from apexdqn.utils.io import ensure_dir
from apexdqn.config import MODELS_DIR, RESULTS_DIR, NOISE_TXT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nn", "xgb"], default="xgb", help="Which baseline to evaluate under attack")
    parser.add_argument("--kmin", type=int, default=1)
    parser.add_argument("--kmax", type=int, default=21)
    parser.add_argument("--xgb_pkl", type=str, default=None)
    args = parser.parse_args()

    ensure_dir(RESULTS_DIR)
    df_train, df_val = load_raw()
    Xtr, ytr, Xva, yva = filter_building_floor(df_train, df_val)
    Xtr = drop_constant_and_weak(Xtr)
    Xva = align_validation_columns(Xtr, Xva)
    Xtr, ytr, Xva, yva = stitch_500_from_train(Xtr, ytr, Xva.values, yva.values)

    # prepare baseline
    if args.backend == "nn":
        from apexdqn.models.nn_model import build_nn
        # If you want to train here, call fit_and_eval in scripts.train_baselines
        # For convenience, try to load a saved model if present
        nn_path = MODELS_DIR / "indoor_nn.keras"
        if nn_path.exists():
            import tensorflow as tf
            nn_model = tf.keras.models.load_model(nn_path)
            # scaler should be saved and loaded as joblib if you saved it; here we assume not
            raise RuntimeError("Saved NN loaded but scaler not provided. Train baselines and save scaler, or run train_baselines first.")
        else:
            raise RuntimeError("No saved NN model found. Run scripts.train_baselines first or use backend=xgb.")
    else:
        # try load pre-trained xgb if provided; else train a fresh xgb by calling training routine
        xgb_pkl = Path(args.xgb_pkl) if args.xgb_pkl else (MODELS_DIR / "xgb_model.pkl")
        if xgb_pkl.exists():
            xgb_model = joblib.load(xgb_pkl)
        else:
            # train a quick xgb and optionally save
            from apexdqn.models.xgb_model import fit_and_eval
            xgb_model, _ = fit_and_eval(Xtr, ytr, Xva, yva, use_pkl=None)
            # joblib.dump(xgb_model, xgb_pkl)

    # noise read
    noise_is_scalar, noise_value = read_noise(NOISE_TXT, n_samples=Xva.shape[0])

    ap_cols = ap_only_columns(Xtr)
    out_path = RESULTS_DIR / "experiment_progress_21APs_unified.txt"
    with open(out_path, "w") as f:
        f.write("Num_Attacked_APs,AP_Combination,MSE\n")

    for k in range(args.kmin, args.kmax + 1):
        if args.backend == "nn":
            model_or_tuple = (nn_model, None)  # NB: scaler required
        else:
            model_or_tuple = xgb_model
        sorted_losses, elapsed = brute_force_evaluate(
            backend=args.backend,
            model_or_tuple=model_or_tuple,
            X_val=Xva,
            y_val=yva,
            ap_cols=ap_cols,
            k=k,
            noise_is_scalar=noise_is_scalar,
            noise_value=noise_value,
            out_fpath=out_path
        )
        top_combo = next(iter(sorted_losses))
        top_mse = sorted_losses[top_combo]
        print(f"k={k} top combo {top_combo} mse={top_mse:.6f} time={elapsed:.2f}s")
    print(f"Wrote results to {out_path}")

if __name__ == "__main__":
    main()
