from pathlib import Path
from apexdqn.data import load_raw
from apexdqn.preprocessing import filter_building_floor, drop_constant_and_weak, align_validation_columns, stitch_500_from_train
from apexdqn.models.nn_model import fit_and_eval as nn_fit
from apexdqn.models.linear_model import fit_and_eval as lr_fit
from apexdqn.models.xgb_model import fit_and_eval as xgb_fit
from apexdqn.config import MODELS_DIR, RESULTS_DIR
from apexdqn.utils.io import ensure_dir

def main():
    ensure_dir(MODELS_DIR); ensure_dir(RESULTS_DIR)
    df_train, df_val = load_raw()
    Xtr, ytr, Xva, yva = filter_building_floor(df_train, df_val)

    Xtr = drop_constant_and_weak(Xtr)
    Xva = align_validation_columns(Xtr, Xva)
    Xtr, ytr, Xva, yva = stitch_500_from_train(Xtr, ytr, Xva.values, yva.values)

    # NN
    nn_model, scaler_out, nn_metrics = nn_fit(Xtr, ytr, Xva, yva)
    # Optional save: nn_model.save(MODELS_DIR / "indoor_nn.keras")

    # Linear
    _, lr_metrics = lr_fit(Xtr, ytr, Xva, yva)

    # XGB
    xgb_model, xgb_metrics = xgb_fit(Xtr, ytr, Xva, yva, use_pkl=None)

    print("Summary metrics:")
    print("NN:", nn_metrics)
    print("Linear:", lr_metrics)
    print("XGB:", xgb_metrics)

if __name__ == "__main__":
    main()
