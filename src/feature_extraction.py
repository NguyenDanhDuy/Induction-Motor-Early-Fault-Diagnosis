# =============================================================================
# STEP 2: FEATURE EXTRACTION
# =============================================================================

window_files = glob.glob(os.path.join(DIRS['windows'], "*.npy"))
print(f"Located {len(window_files)} window files.")

X_list = []
y_list = []

print("Extracting features (11 statistics x 6 sensors)...")
for w_file in tqdm(window_files, desc="Extracting"):
    
    content = np.load(w_file, allow_pickle=True).item()
    windows = content['data'] # (N, 132, 6)
    label = content['label']

    file_feats = []
    
    for win in windows:
        win_feat_vec = []
        # Process each of the 6 sensor channels
        for col_idx in range(6):
            col_signal = win[:, col_idx]
            feats = extract_features_11(col_signal)
            win_feat_vec.extend(feats)
        file_feats.append(win_feat_vec)

    X_list.extend(file_feats)
    y_list.extend([label] * len(file_feats))

    # Memory Cleanup
    del content, windows, file_feats
    gc.collect()

X_raw = np.array(X_list)
y_raw = np.array(y_list)

print(f"Feature Matrix Dimensions: {X_raw.shape}") # (Total_Samples, 66)
print(f"Labels Dimensions: {y_raw.shape}")

# Save raw features
np.save(os.path.join(DIRS['features'], 'X_raw.npy'), X_raw)
np.save(os.path.join(DIRS['features'], 'y_raw.npy'), y_raw)

print(f"Step 2 Completed. Raw features saved.")
del X_list, y_list, X_raw, y_raw
gc.collect()