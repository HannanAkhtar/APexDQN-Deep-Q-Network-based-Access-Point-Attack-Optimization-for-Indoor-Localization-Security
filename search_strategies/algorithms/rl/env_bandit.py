import numpy as np
from copy import deepcopy


class BanditEnv:
    """
    One-step contextual bandit environment for DQN-based AP subset selection.
    """

    def __init__(self, baseline_backend, baseline_model, X_val, y_val, attack_combinations, noise_is_scalar, noise_value):
        self.baseline_backend = baseline_backend
        self.baseline_model = baseline_model
        self.X_val = X_val
        self.y_val = y_val
        self.attack_combinations = list(attack_combinations)
        self.noise_is_scalar = noise_is_scalar
        self.noise_value = noise_value

    def reset(self):
        return np.zeros(len(self.attack_combinations), dtype=np.float32)

    def step(self, action_index):
        combo = self.attack_combinations[action_index]
        X_att = deepcopy(self.X_val)
        if self.noise_is_scalar:
            for c in combo:
                X_att[c] = X_att[c] + self.noise_value
        else:
            for c in combo:
                X_att[c] = X_att[c] + self.noise_value

        if self.baseline_backend == 'nn':
            nn_model, scaler_out = self.baseline_model
            preds_s = nn_model.predict(X_att, verbose=0)
            preds = scaler_out.inverse_transform(preds_s)
        else:
            xgb_model = self.baseline_model
            preds = xgb_model.predict(X_att)

        mse = float(np.mean((preds - self.y_val.values) ** 2))
        reward = mse
        done = True
        next_state = np.zeros(len(self.attack_combinations), dtype=np.float32)
        return next_state, reward, done
