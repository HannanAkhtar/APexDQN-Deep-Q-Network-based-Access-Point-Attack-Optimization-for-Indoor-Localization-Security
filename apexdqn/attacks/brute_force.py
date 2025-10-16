"""
Unified brute-force attack code for both NN and XGB backends.
This preserves the exact behavior and log format of your notebooks:
- loops k=1..21
- enumerates combinations of AP columns (exclude last 4 non-AP columns)
- applies scalar or per-sample noise
- computes MSE on original scale and writes lines:
   Num_Attacked_APs,"AP1,AP2,...",MSE
- then writes Time_Taken_sec: <seconds>
"""
import time
from itertools import combinations
from tqdm import tqdm
import numpy as np
from pathlib import Path

def apply_noise(X_val_df, combo, noise_is_scalar, noise_value, noise_array=None):
    X_att = X_val_df.copy()
    for col in combo:
        if noise_is_scalar:
            X_att[col] = X_att[col] + noise_value
        else:
            X_att[col] = X_att[col] + noise_array
    return X_att

def brute_force_evaluate(
    backend,                # "nn" or "xgb"
    model_or_tuple,        # for nn: (nn_model, scaler_out) ; for xgb: xgb_model
    X_val, y_val,
    ap_cols,               # iterable of AP column names
    k,
    noise_is_scalar,
    noise_value,
    out_fpath: Path,
):
    """
    Evaluate all combinations of size k; return sorted dict {combo_tuple: mse}
    and write to out_fpath (append).
    """
    combos = list(combinations(ap_cols, k))
    # timing
    start_time = time.time()
    losses = {}

    for combo in tqdm(combos, desc=f"Brute forcing k={k}", leave=False):
        X_att = apply_noise(X_val, combo, noise_is_scalar, noise_value, noise_array=(None if noise_is_scalar else noise_value))
        if backend == "nn":
            nn_model, scaler_out = model_or_tuple
            preds_s = nn_model.predict(X_att, verbose=0)
            preds = scaler_out.inverse_transform(preds_s)
        else:  # xgb
            xgb_model = model_or_tuple
            preds = xgb_model.predict(X_att)

        mse = float(np.mean((preds - y_val.values)**2))
        losses[combo] = mse

    # sort descending (attack aims to increase MSE)
    sorted_losses = dict(sorted(losses.items(), key=lambda kv: kv[1], reverse=True))
    elapsed = time.time() - start_time

    # write results
    with open(out_fpath, "a") as f:
        for key, value in sorted_losses.items():
            combo_str = ",".join(key)
            f.write(f"{k},\"{combo_str}\",{value}\n")
        f.write(f"Time_Taken_sec: {elapsed:.2f}\n\n")

    return sorted_losses, elapsed
