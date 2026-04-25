# -*- coding: utf-8 -*-
"""
Central configuration settings for the Wi-Fi fingerprinting benchmark suite.

This module contains all static parameters, file paths, and hyperparameters
used throughout the search-strategy benchmarks.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'
FIGURES_DIR = ROOT_DIR / 'figures'

TRAIN_CSV = DATA_DIR / 'TrainingData.csv'
VAL_CSV = DATA_DIR / 'ValidationData.csv'
NOISE_TXT = DATA_DIR / 'noise.txt'

BUILDING_ID = 2
FLOOR = 3
NUM_NON_AP_FEATURES = 4

NN_EPOCHS = 250
NN_BATCH_SIZE = 32
L2_REG = 0.01
DROPOUT_RATE = 0.3

RANDOM_SEED = 42
