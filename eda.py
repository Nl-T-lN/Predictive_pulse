"""
=============================================================================
PHASE 2: Exploratory Data Analysis (EDA)
Hypertension Prediction System
=============================================================================
Generates 7 key visualizations and saves them as PNG files.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs('eda_plots', exist_ok=True)

# Load raw data and apply same cleaning (pre-encoding for meaningful labels)
data = pd.read_csv('patient_data.csv')
data.rename(columns={'C': 'Gender'}, inplace=True)
data['TakeMedication'] = data['TakeMedication'].str.strip()
data['NoseBleeding'] = data['NoseBleeding'].str.strip()
data['Systolic'] = data['Systolic'].replace({'121- 130': '121 - 130', '100+': '100 - 110'})
data['Stages'] = data['Stages'].replace({
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)',
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS'
})
data['Diastolic'] = data['Diastolic'].replace({'130+': '100+'})
data.drop_duplicates(inplace=True)

print(f"Clean data loaded: {data.shape[0]} rows")
print("Generating visualizations...\n")

# ---- 1. Gender Distribution (Count Plot) ----
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Gender", palette="Set2")
plt.title("Gender Distribution")
plt.savefig('eda_plots/01_gender_countplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 1/7 Gender Distribution (count plot) saved")

# ---- 2. Gender Distribution (Pie Chart) ----
plt.figure(figsize=(5, 5))
data['Gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#99ff99'])
plt.title("Gender Distribution (Pie Chart)")
plt.ylabel("")
plt.savefig('eda_plots/02_gender_pie.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 2/7 Gender Distribution (pie chart) saved")

# ---- 3. Hypertension Stages Distribution ----
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Stages", palette="coolwarm")
plt.title("Hypertension Stages Distribution")
plt.xticks(rotation=30)
plt.savefig('eda_plots/03_stages_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 3/7 Hypertension Stages Distribution saved")

# ---- 4. Correlation Heatmap (Systolic vs Diastolic) ----
# Convert BP ranges to midpoints for correlation
systolic_map = {'100 - 110': 105, '111 - 120': 115.5, '121 - 130': 125.5, '130+': 140}
diastolic_map = {'70 - 80': 75, '81 - 90': 85.5, '91 - 100': 95.5, '100+': 110}
data_numeric = data.copy()
data_numeric['Systolic_mid'] = data_numeric['Systolic'].map(systolic_map)
data_numeric['Diastolic_mid'] = data_numeric['Diastolic'].map(diastolic_map)

plt.figure(figsize=(6, 5))
corr = data_numeric[['Systolic_mid', 'Diastolic_mid']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title("Correlation: Systolic vs Diastolic Blood Pressure")
plt.savefig('eda_plots/04_bp_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 4/7 Correlation Heatmap saved")

# ---- 5. TakeMedication vs Severity ----
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Severity", hue="TakeMedication", palette="Set1")
plt.title("TakeMedication vs Severity")
plt.savefig('eda_plots/05_medication_vs_severity.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 5/7 TakeMedication vs Severity saved")

# ---- 6. Age Group vs Hypertension Stages ----
plt.figure(figsize=(10, 5))
sns.countplot(data=data, x="Age", hue="Stages", palette="viridis")
plt.title("Age Group vs Hypertension Stages")
plt.savefig('eda_plots/06_age_vs_stages.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 6/7 Age Group vs Hypertension Stages saved")

# ---- 7. Pairplot: Systolic vs Diastolic by Stages ----
pair_data = data_numeric[['Systolic_mid', 'Diastolic_mid', 'Stages']].dropna()
g = sns.pairplot(pair_data, hue="Stages", palette="Set1", diag_kind='kde')
g.figure.suptitle("Pairplot: Systolic vs Diastolic across Stages", y=1.02)
g.savefig('eda_plots/07_bp_pairplot.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("[OK] 7/7 Pairplot saved")

print(f"\n{'=' * 60}")
print("PHASE 2 COMPLETE - All 7 EDA plots saved in eda_plots/")
print(f"{'=' * 60}")
