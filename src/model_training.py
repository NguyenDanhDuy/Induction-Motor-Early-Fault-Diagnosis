# =============================================================================
# STEP 4: MODEL TRAINING & EVALUATION
# =============================================================================

# Load prepared data
X_train = np.load(os.path.join(DIRS['ready'], 'X_train.npy'))
y_train = np.load(os.path.join(DIRS['ready'], 'y_train.npy'))
X_test = np.load(os.path.join(DIRS['ready'], 'X_test.npy'))
y_test = np.load(os.path.join(DIRS['ready'], 'y_test.npy'))

# Initialize and train SVM with RBF kernel and One-vs-One decision function
print("Training SVM Classifier (RBF Kernel)...")
svm = SVC(kernel='rbf', decision_function_shape='ovo', C=10, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
print("Model Training Complete.")

# Save trained model
joblib.dump(svm, os.path.join(DIRS['models'], 'svm_model.pkl'))

# Model Evaluation
print("\nEvaluating Model Performance...")

# Training Accuracy
y_train_pred = svm.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# Testing Accuracy
y_pred = svm.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("="*40)
print(f"FINAL RESULTS:")
print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Macro F1-Score:    {f1:.4f}")
print("="*40)

# Overfitting Analysis
gap = train_acc - test_acc
if gap > 0.05:
    print(f"Warning: Potential overfitting detected (Difference: {gap:.2%})")
elif gap < -0.02:
    print(f"Note: Testing accuracy exceeds training accuracy.")
else:
    print("Model demonstrates good fit and generalization.")

# Classification Report
target_names = ['Normal', 'Imbalance', 'Misalignment', 'Bearing Fault']
print("\n" + classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(DIRS['results'], 'confusion_matrix.png'))
plt.show()

print("\nPipeline Execution Completed Successfully.")