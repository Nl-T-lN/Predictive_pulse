"""
=============================================================================
PHASE 1: Data Preparation & Scaler Pipeline
Hypertension Prediction System
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ---- 1.1 Load & Inspect ----
print("=" * 60)
print("PHASE 1: DATA PREPARATION & SCALER PIPELINE")
print("=" * 60)

data = pd.read_csv('patient_data.csv')
print(f"\n[OK] Dataset loaded: {data.shape[0]} rows x {data.shape[1]} columns")
print(f"\nFirst 5 rows:\n{data.head()}")
print(f"\nNull values per column:\n{data.isnull().sum()}")

# ---- 1.2 Column Rename ----
data.rename(columns={'C': 'Gender'}, inplace=True)
print("\n[OK] Column 'C' renamed to 'Gender'")

# ---- 1.3 Inconsistency Corrections ----
data['TakeMedication'].replace({'Yes ': 'Yes'}, inplace=True)
data['NoseBleeding'].replace({'No ': 'No'}, inplace=True)
data['Systolic'].replace({'121- 130': '121 - 130'}, inplace=True)
data['Systolic'].replace({'100+': '100 - 110'}, inplace=True)
data['Stages'].replace({'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'}, inplace=True)
data['Stages'].replace({'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS'}, inplace=True)

print(f"\nDiastolic == '130+' count: {(data['Diastolic'] == '130+').sum()}")
print(f"Diastolic == '100+' count: {(data['Diastolic'] == '100+').sum()}")
data['Diastolic'].replace({'130+': '100+'}, inplace=True)

print("[OK] Inconsistencies corrected")

# ---- 1.4 Duplicate Removal ----
dup_count = data.duplicated().sum()
print(f"\n[INFO] Duplicates found: {dup_count}")
data.drop_duplicates(inplace=True)
print(f"[OK] Duplicates removed. Clean dataset: {data.shape[0]} rows x {data.shape[1]} columns")

# ---- 1.5 Label Encoding ----
nominal_features = ['Gender', 'History', 'Patient', 'TakeMedication',
                    'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet']
ordinal_features = [f for f in data.columns if f not in nominal_features]
ordinal_features.remove('Stages')

print(f"\nNominal features: {nominal_features}")
print(f"Ordinal features: {ordinal_features}")

# Encode nominal (binary) features
for col in nominal_features:
    if set(data[col].unique()) == set(['Yes', 'No']):
        data[col] = data[col].map({'No': 0, 'Yes': 1})
    elif col == 'Gender':
        data[col] = data[col].map({'Male': 0, 'Female': 1})

# Encode ordinal features
data['Age'] = data['Age'].map({'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4})
data['Severity'] = data['Severity'].replace({'Mild': 0, 'Moderate': 1, 'Sever': 2})
data['Whendiagnoused'] = data['Whendiagnoused'].map({'<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3})
data['Systolic'] = data['Systolic'].map({'100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3})
data['Diastolic'] = data['Diastolic'].map({'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3})

# Encode target variable
data['Stages'] = data['Stages'].map({
    'NORMAL': 0,
    'HYPERTENSION (Stage-1)': 1,
    'HYPERTENSION (Stage-2)': 2,
    'HYPERTENSIVE CRISIS': 3
})

print("[OK] Label encoding applied")
print(f"\nEncoded data sample:\n{data.head()}")
print(f"\nData types:\n{data.dtypes}")

# Check for NaN from failed mappings
nan_check = data.isnull().sum()
if nan_check.sum() > 0:
    print(f"\n[WARNING] NaN values detected after encoding:\n{nan_check[nan_check > 0]}")
    data.dropna(inplace=True)
    print(f"Dropped NaN rows. Remaining: {data.shape[0]} rows")
else:
    print("\n[OK] No NaN values after encoding - all mappings successful")

# ---- 1.6 Feature Scaling ----
scaler = MinMaxScaler()
data[ordinal_features] = scaler.fit_transform(data[ordinal_features])

print(f"\n[OK] MinMaxScaler applied to: {ordinal_features}")
print(f"\nScaled data sample:\n{data.head()}")

# ---- 1.7 Save artifacts ----
joblib.dump(scaler, 'scaler.pkl')
print("\n[OK] Scaler saved as 'scaler.pkl'")

data.to_csv('processed_data.csv', index=False)
print("[OK] Processed data saved as 'processed_data.csv'")

print(f"\n{'=' * 60}")
print(f"PHASE 1 COMPLETE")
print(f"Final dataset: {data.shape[0]} rows x {data.shape[1]} columns")
print(f"Target distribution:")
print(data['Stages'].value_counts().sort_index())
print(f"{'=' * 60}")
