"""
=============================================================================
PHASE 3: Model Training & Evaluation
Hypertension Prediction System
=============================================================================
Trains Logistic Regression on processed data, evaluates, and saves model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---- 3.1 Load Processed Data ----
print("=" * 60)
print("PHASE 3: MODEL TRAINING & EVALUATION")
print("=" * 60)

data = pd.read_csv('processed_data.csv')
print(f"\n[OK] Processed data loaded: {data.shape[0]} rows x {data.shape[1]} columns")

# ---- 3.2 Train/Test Split ----
X = data.drop('Stages', axis=1)
y = data['Stages']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# ---- 3.3 Train Logistic Regression ----
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
print("\n[OK] Logistic Regression model trained")

# ---- 3.4 Evaluate ----
y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- EVALUATION RESULTS ---")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

print(f"\nClassification Report:")
stage_names = ['NORMAL (0)', 'Stage-1 (1)', 'Stage-2 (2)', 'CRISIS (3)']
report = classification_report(y_test, y_pred, target_names=stage_names)
print(report)

print(f"Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ---- 3.5 Save Model ----
joblib.dump(logreg, 'logreg_model.pkl')
print(f"\n[OK] Model saved as 'logreg_model.pkl'")

# ---- 3.6 Also save the scaler alongside (verify it exists) ----
try:
    scaler = joblib.load('scaler.pkl')
    print("[OK] scaler.pkl verified and accessible")
except FileNotFoundError:
    print("[WARNING] scaler.pkl not found! Run data_preparation.py first.")

print(f"\n{'=' * 60}")
print("PHASE 3 COMPLETE")
print(f"Model: Logistic Regression (accuracy={accuracy*100:.1f}%)")
print(f"{'=' * 60}")
