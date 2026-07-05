# =============================================================================
# 2. FILE SELECTION LOGIC & UTILITIES
# =============================================================================

def get_csvs_in_dir(directory):
    """Retrieve all CSV files within a specific directory (non-recursive)."""
    return glob.glob(os.path.join(directory, "*.csv"))

def select_training_files(base_path):
    """
    Selects a stratified subset of files from the MAFAULDA dataset to ensure balanced class representation and manage memory constraints.
    - Normal: 20 files.
    - Imbalance: 21 files (3 from each of the 7 weight configurations).
    - Misalignment: 20 files (2 from each horizontal/vertical configuration).
    - Bearing: 24 files (randomly sampled from outer, ball, and cage fault subfolders).
    """
    selected_files = [] # List of (filepath, label)

    # --- CLASS 0: NORMAL (20 files) ---
    normal_path = os.path.join(base_path, 'normal')
    all_normal = glob.glob(os.path.join(normal_path, "**/*.csv"), recursive=True)
    selected = random.sample(all_normal, min(len(all_normal), 20))
    selected_files.extend([(f, 0) for f in selected])
    print(f"Normal selected: {len(selected)}")

    # --- CLASS 1: IMBALANCE (7 folders x 3 files = 21) ---
    imb_path = os.path.join(base_path, 'imbalance')
    imb_subs = [f.path for f in os.scandir(imb_path) if f.is_dir()]
    count = 0
    for sub in imb_subs:
        csvs = get_csvs_in_dir(sub)
        chosen = random.sample(csvs, min(len(csvs), 3))
        selected_files.extend([(f, 1) for f in chosen])
        count += len(chosen)
    print(f"Imbalance selected: {count}")

    # --- CLASS 2: MISALIGNMENT (10 folders x 2 files = 20) ---
    mis_count = 0
    for m_type in ['horizontal-misalignment', 'vertical-misalignment']:
        m_path = os.path.join(base_path, m_type)
        if not os.path.exists(m_path): continue
        subs = [f.path for f in os.scandir(m_path) if f.is_dir()]
        for sub in subs:
            csvs = get_csvs_in_dir(sub)
            chosen = random.sample(csvs, min(len(csvs), 2))
            selected_files.extend([(f, 2) for f in chosen])
            mis_count += len(chosen)
    print(f"Misalignment selected: {mis_count}")

    # --- CLASS 3: BEARING FAULT (~24 files) ---
    bear_count = 0
    bearing_roots = ['underhang', 'overhang']
    fault_types = ['outer_race', 'ball_fault', 'cage_fault']

    leaf_folders = []
    for root in bearing_roots:
        for ftype in fault_types:
            path = os.path.join(base_path, root, ftype)
            if os.path.exists(path):
                weights = [f.path for f in os.scandir(path) if f.is_dir()]
                leaf_folders.extend(weights)

    random.shuffle(leaf_folders)

    current_bear_files = []
    for leaf in leaf_folders:
        csvs = get_csvs_in_dir(leaf)
        chosen = random.sample(csvs, min(len(csvs), 2))
        current_bear_files.extend(chosen)

    final_bearing = random.sample(current_bear_files, min(len(current_bear_files), 24))
    selected_files.extend([(f, 3) for f in final_bearing])
    print(f"Bearing selected: {len(final_bearing)}")

    print(f"TOTAL TRAINING FILES: {len(selected_files)}")
    return selected_files

def extract_features_11(signal):
    """
    Extract 11 time-domain statistical features from a 1D signal array.
    Features: Mean, Min, Max, Standard Deviation, Q1, Median (Q2), Q3, Kurtosis, Skewness, RMS, Energy.
    """
    return [
        np.mean(signal),              # 1. Mean
        np.min(signal),               # 2. Min
        np.max(signal),               # 3. Max
        np.std(signal),               # 4. Std
        np.percentile(signal, 25),    # 5. Q1
        np.median(signal),            # 6. Q2
        np.percentile(signal, 75),    # 7. Q3
        stats.kurtosis(signal),       # 8. Kurtosis
        stats.skew(signal),           # 9. Skewness
        np.sqrt(np.mean(signal**2)),  # 10. RMS
        np.sum(signal**2)             # 11. Energy
    ]