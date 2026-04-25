import numpy as np
from tensorflow.keras import Sequential, layers
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from .config import L2_REG, DROPOUT_RATE, NN_EPOCHS, NN_BATCH_SIZE

def build_nn(input_dim):
    """Constructs and compiles the baseline neural network model.

    Args:
        input_dim (int): The number of input features for the first layer.

    Returns:
        tensorflow.keras.Model: The compiled Keras model.
    """
    model = Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(L2_REG)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(L2_REG)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(64, activation='relu', kernel_regularizer=l2(L2_REG)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
        layers.Dense(2)  # Output layer for longitude and latitude.
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def fit_and_eval(X_train, y_train, X_val, y_val):
    """Trains the NN model, evaluates it, and returns the results.

    This function handles scaling of target variables, model training, prediction,
    and inverse-scaling to report metrics in their original units (meters).

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.DataFrame): Training target data.
        X_val (pd.DataFrame): Validation feature data.
        y_val (pd.DataFrame): Validation target data.

    Returns:
        tuple: A tuple containing:
            - model (tensorflow.keras.Model): The trained model.
            - scaler_output (StandardScaler): The fitted scaler for target variables.
            - results (dict): A dictionary with performance metrics and training history.
    """
    # Scale target variables for stable training.
    scaler_output = StandardScaler()
    ytr_s = scaler_output.fit_transform(y_train)
    yva_s = scaler_output.transform(y_val)

    # Build and train the model.
    model = build_nn(X_train.shape[1])
    history = model.fit(X_train, ytr_s, epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE,
                        validation_data=(X_val, yva_s), verbose=1)

    # Evaluate and inverse-transform predictions to original scale.
    preds_s = model.predict(X_val, verbose=0)
    preds = scaler_output.inverse_transform(preds_s)

    mse_org = float(np.mean((preds - y_val.values)**2))
    mae_org = float(np.mean(np.abs(preds - y_val.values)))
    print(f"Original-scale Validation MSE: {mse_org:.6f}, MAE: {mae_org:.6f}")

    results = dict(mse=mse_org, mae=mae_org, history=history.history)
    return model, scaler_output, results

