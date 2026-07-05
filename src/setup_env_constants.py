import os
import glob
import random
import gc
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler

# =============================================================================
# 1. SETUP ENVIRONMENT & CONSTANTS
# =============================================================================

# --- CONFIGURATION ---
BASE_DATA_PATH = '/content/drive/MyDrive/mafaulda'
OUTPUT_BASE_PATH = '/content/drive/MyDrive/data' 
# Link to drive folder where the data is stored and where the output will be saved. Change this path as per your environment.
DIRS = {
    'windows': os.path.join(OUTPUT_BASE_PATH, 'step1_windows'),
    'features': os.path.join(OUTPUT_BASE_PATH, 'step2_features'),
    'eda': os.path.join(OUTPUT_BASE_PATH, 'plots'),
    'ready': os.path.join(OUTPUT_BASE_PATH, 'step4_ready'),
    'models': os.path.join(OUTPUT_BASE_PATH, 'models'),
    'results': os.path.join(OUTPUT_BASE_PATH, 'results')
}

# Ensure directories exist
for path in DIRS.values():
    os.makedirs(path, exist_ok=True)
    print(f"Verified directory: {path}")

# Signal processing parameters
WINDOW_SIZE = 132
OVERLAP = 0.25
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP)) # ~99 samples
TARGET_COLS_INDICES = [1, 2, 3, 4, 5, 6]  # Sensor columns 1 to 6 (Excluding tachometer and microphone)

print(f"\nEnvironment Configured Successfully. Stride calculated as: {STRIDE}")
