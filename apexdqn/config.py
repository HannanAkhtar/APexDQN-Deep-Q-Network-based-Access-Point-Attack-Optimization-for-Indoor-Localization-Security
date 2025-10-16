# -*- coding: utf-8 -*-
"""
Central configuration settings for the APexDQN project.

This module contains all static parameters, file paths, and hyperparameters
to ensure experimental consistency and reproducibility.
"""

from pathlib import Path

# --- Path Definitions ---
# Project directory structure. Enables OS-agnostic paths.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# --- Data Source Files ---
# Explicit paths to the required dataset files.
TRAIN_CSV = DATA_DIR / "TrainingData.csv"
VAL_CSV   = DATA_DIR / "ValidationData.csv"

# --- Dataset Filtering ---
# Defines the specific data subset for all experiments.
BUILDING_ID = 2
FLOOR = 3

# --- Feature Engineering ---
# Number of trailing columns in the dataset that are not Access Points.
NUM_NON_AP_FEATURES = 4

# --- Neural Network Hyperparameters ---
# Defines the training configuration for the baseline NN model.
NN_EPOCHS = 250
NN_BATCH_SIZE = 32
L2_REG = 0.01
DROPOUT_RATE = 0.3

# --- Reproducibility ---
# Global random seed for deterministic and reproducible results.
RANDOM_SEED = 42

