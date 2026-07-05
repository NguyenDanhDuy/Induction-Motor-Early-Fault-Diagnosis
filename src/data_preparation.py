# =============================================================================
# STEP 3: DATA PREPARATION (SPLIT -> SCALE -> OVERSAMPLE)
# =============================================================================

# Load raw features and labels
X = np.load(os.path.join(DIRS['features'], 'X_raw.npy'))
y = np.load(os.path.join(DIRS['features'], 'y_raw.npy'))

# Train-test split (70/30) using stratified sampling
print("Splitting Data...")
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(splitter.split(X, y))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

del X, y
gc.collect()

# Standardize features (Fit on training data ONLY to prevent data leakage)
print("Scaling Data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(DIRS['models'], 'scaler.pkl'))

# Apply Random Oversampling exclusively to the training set to address class imbalance
print(f"Original Training Set Dimensions: {X_train.shape}")
print("Oversampling...")
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
print(f"Balanced Training Set Dimensions: {X_train_res.shape}")

# Save prepared datasets
np.save(os.path.join(DIRS['ready'], 'X_train.npy'), X_train_res)
np.save(os.path.join(DIRS['ready'], 'y_train.npy'), y_train_res)
np.save(os.path.join(DIRS['ready'], 'X_test.npy'), X_test)
np.save(os.path.join(DIRS['ready'], 'y_test.npy'), y_test)

print("Step 3 Completed. Prepared data saved for model training.")
del X_train, X_test, X_train_res, y_train_res
gc.collect()