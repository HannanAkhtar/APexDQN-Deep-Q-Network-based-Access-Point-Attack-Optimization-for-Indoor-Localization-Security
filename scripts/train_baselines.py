from pathlib import Path
import joblib
from search_strategies.data import load_raw
from search_strategies.preprocessing import filter_building_floor, drop_constant_and_weak, align_validation_columns, stitch_500_from_train
from search_strategies.models.nn_model import fit_and_eval as nn_fit
from search_strategies.models.linear_model import fit_and_eval as lr_fit
from search_strategies.models.xgb_model import fit_and_eval_xgb as xgb_fit
from search_strategies.config import MODELS_DIR, RESULTS_DIR
from search_strategies.utils.io import ensure_dir


def main():
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    df_train, df_val = load_raw()
    Xtr, ytr, Xva, yva = filter_building_floor(df_train, df_val)

    Xtr = drop_constant_and_weak(Xtr)
    Xva = align_validation_columns(Xtr, Xva)
    Xtr, ytr, Xva, yva = stitch_500_from_train(Xtr, ytr, Xva.values, yva.values)

    nn_model, scaler_out, nn_metrics = nn_fit(Xtr, ytr, Xva, yva)
    nn_model.save(MODELS_DIR / 'indoor_nn.keras')
    joblib.dump(scaler_out, MODELS_DIR / 'scaler_output.joblib')

    _, lr_metrics = lr_fit(Xtr, ytr, Xva, yva)

    xgb_model, xgb_metrics = xgb_fit(Xtr, ytr, Xva, yva, model_path=None)
    joblib.dump(xgb_model, MODELS_DIR / 'xgb_model.pkl')

    print('Summary metrics:')
    print('NN:', nn_metrics)
    print('Linear:', lr_metrics)
    print('XGB:', xgb_metrics)


if __name__ == '__main__':
    main()
