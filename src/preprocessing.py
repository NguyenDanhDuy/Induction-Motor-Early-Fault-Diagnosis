# =============================================================================
# STEP 1: PREPROCESSING & WINDOWING
# =============================================================================

training_files = select_training_files(BASE_DATA_PATH)

print("\nStarting Windowing Process...")

for i, (filepath, label) in enumerate(tqdm(training_files, desc="Processing CSVs")):
    try:
        df = pd.read_csv(filepath)

        data = df.iloc[:, TARGET_COLS_INDICES].values

        windows = []
        num_samples = data.shape[0]
        for start in range(0, num_samples - WINDOW_SIZE + 1, STRIDE):
            win = data[start : start + WINDOW_SIZE, :]
            windows.append(win)

        if not windows: continue

        windows_np = np.array(windows) # Shape (N, 132, 6)

        # Save to disk to manage memory constraints
        save_name = f"win_{i}_label_{label}.npy"
        np.save(os.path.join(DIRS['windows'], save_name), {'data': windows_np, 'label': label})

        # Memory Cleanup
        del df, data, windows, windows_np
        gc.collect()

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

print(f"Step 1 Completed. Windowed data saved to {DIRS['windows']}")