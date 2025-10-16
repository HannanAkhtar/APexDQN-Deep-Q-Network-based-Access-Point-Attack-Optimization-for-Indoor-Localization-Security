import pandas as pd
from .config import TRAIN_CSV, VAL_CSV

def load_raw():
    """Loads the raw training and validation datasets from source CSV files.

    Returns:
        tuple: A tuple containing the training and validation pandas DataFrames.
    """
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    print("Shape of train set: ", df_train.shape)
    print("Shape of validation set: ", df_val.shape)
    return df_train, df_val

