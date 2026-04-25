import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import BUILDING_ID, FLOOR, NUM_NON_AP_FEATURES, RANDOM_SEED


def filter_building_floor(df_train, df_val):
    """Filters data for a specific building and floor, then separates features and targets."""
    df_train_f = df_train[(df_train['BUILDINGID'] == BUILDING_ID) & (df_train['FLOOR'] == FLOOR)]
    df_val_f = df_val[(df_val['BUILDINGID'] == BUILDING_ID) & (df_val['FLOOR'] == FLOOR)]

    drop_cols = ['LONGITUDE', 'LATITUDE', 'USERID', 'PHONEID', 'TIMESTAMP']
    Xtr = df_train_f.drop(columns=drop_cols)
    ytr = df_train_f[['LONGITUDE', 'LATITUDE']]
    Xva = df_val_f.drop(columns=drop_cols)
    yva = df_val_f[['LONGITUDE', 'LATITUDE']]

    print('X_train_filtered shape:', Xtr.shape)
    print('y_train_filtered shape:', ytr.shape)
    print('X_val_filtered shape:', Xva.shape)
    print('y_val_filtered shape:', yva.shape)
    return Xtr, ytr, Xva, yva


def drop_constant_and_weak(X_train, weak_signal_threshold=-90):
    """Performs feature selection by removing undetected or weak-signal APs."""
    cols_all_100 = X_train.columns[(X_train == 100).all()]
    print('Dropping undetected APs:', cols_all_100.tolist())
    X_train = X_train.drop(cols_all_100, axis=1)

    cols_not_all_100 = X_train.columns[~(X_train == 100).all()]
    cols_to_drop = []
    for c in cols_not_all_100:
        detected = X_train[c][X_train[c] != 100]
        if not detected.empty and detected.max() < weak_signal_threshold:
            cols_to_drop.append(c)
    print('Dropping weak-signal APs:', cols_to_drop)
    X_train = X_train.drop(cols_to_drop, axis=1)

    print(f'Shape after feature selection: {X_train.shape}')
    return X_train


def align_validation_columns(X_train_filtered, X_val):
    """Ensures the validation feature set has the same columns as the training set."""
    X_val = X_val[X_train_filtered.columns]
    print(f'Aligned validation shape: {X_val.shape}')
    return X_val


def augment_validation_set(X_train, y_train, X_val, y_val, size=500):
    """Augments the validation set by transferring samples from the training set."""
    X_train_new, X_take, y_train_new, y_take = train_test_split(
        X_train, y_train, test_size=size, random_state=RANDOM_SEED, shuffle=True
    )

    X_val_new = pd.DataFrame(np.concatenate((X_val, X_take)), columns=X_train_new.columns)
    y_val_new = pd.DataFrame(np.concatenate((y_val, y_take)), columns=y_train_new.columns)

    print(f'Augmented validation set; new shapes: Train={X_train_new.shape}, Val={X_val_new.shape}')
    return X_train_new, y_train_new, X_val_new, y_val_new


def get_ap_columns(X):
    """Returns a list of column names corresponding to Access Points (APs)."""
    return list(X.columns[:-NUM_NON_AP_FEATURES])


# Backwards-compatible aliases used by existing scripts.
ap_only_columns = get_ap_columns
stitch_500_from_train = augment_validation_set
