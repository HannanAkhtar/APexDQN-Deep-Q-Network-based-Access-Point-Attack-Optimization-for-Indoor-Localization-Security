import numpy as np
from copy import deepcopy

class BanditEnv:
    """
    One-step contextual bandit environment for AP subset selection.
    The environment returns a zero state vector (len = num_actions), and
    the reward is computed as the MSE of the baseline model on attacked data.
    """
    def __init__(self, baseline_backend, baseline_model, X_val, y_val, attack_combinations, noise_is_scalar, noise_value):
        """
        baseline_backend: 'nn' or 'xgb'
        baseline_model: for nn -> tuple(nn_model, scaler_out), for xgb -> model
        attack_combinations: list of tuples of AP column names (ordered)
        noise_*: noise specification (scalar or per-sample array)
        """
        self.baseline_backend = baseline_backend
        self.baseline_model = baseline_model
        self.X_val = X_val
        self.y_val = y_val
        self.attack_combinations = list(attack_combinations)
        self.noise_is_scalar = noise_is_scalar
        self.noise_value = noise_value

    def reset(self):
        # state is zero-vector (as in paper), length = number of actions
        return np.zeros(len(self.attack_combinations), dtype=np.float32)

    def step(self, action_index):
        combo = self.attack_combinations[action_index]
        X_att = deepcopy(self.X_val)
        # apply noise
        if self.noise_is_scalar:
            for c in combo:
                X_att[c] = X_att[c] + self.noise_value
        else:
            for c in combo:
                X_att[c] = X_att[c] + self.noise_value  # noise_value is array per-sample

        if self.baseline_backend == "nn":
            nn_model, scaler_out = self.baseline_model
            preds_s = nn_model.predict(X_att, verbose=0)
            preds = scaler_out.inverse_transform(preds_s)
        else:
            xgb_model = self.baseline_model
            preds = xgb_model.predict(X_att)

        mse = float(np.mean((preds - self.y_val.values)**2))
        reward = mse  # immediate reward (paper uses MSE increase; when using frozen baseline, you can subtract original mse if desired)
        done = True  # one-step episode
        next_state = np.zeros(len(self.attack_combinations), dtype=np.float32)
        return next_state, reward, done
