# %% [markdown]
# # Part 2: Data Preparation (Do Once, Reuse Across All Methods)
# This script downloads the datasets and performs minimal, shared preprocessing to 
# ensure all models are evaluated on the exact same train/test splits.

# %%
import pandas as pd
import numpy as np
import os
import kaggle
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# Ensure data directory exists
os.makedirs('../data', exist_ok=True)

# %% [markdown]
# ### Step 1: Telco Churn Preprocessing
# Source: Kaggle (blastchar/telco-customer-churn)

# %%
print("Downloading Telco Customer Churn dataset...")
kaggle.api.dataset_download_files('blastchar/telco-customer-churn', path='../data', unzip=True)
df_telco = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 1. Fix TotalCharges column (stored as string, spaces for missing values)
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')
df_telco['TotalCharges'] = df_telco['TotalCharges'].fillna(0)

# 2. Encode target column Churn as binary (Yes=1, No=0)
df_telco['Churn'] = df_telco['Churn'].map({'Yes': 1, 'No': 0})

# 3. Drop customerID
df_telco = df_telco.drop(columns=['customerID'])

# Save minimally-processed version as shared base (optional but good practice)
df_telco.to_csv('../data/telco_churn_base.csv', index=False)

# 4 & 5. Stratified 80/20 train/test split
X_telco = df_telco.drop(columns=['Churn'])
y_telco = df_telco['Churn']
X_train_telco, X_test_telco, y_train_telco, y_test_telco = train_test_split(
    X_telco, y_telco, test_size=0.2, stratify=y_telco, random_state=42
)

train_telco = pd.concat([X_train_telco, y_train_telco], axis=1)
test_telco = pd.concat([X_test_telco, y_test_telco], axis=1)

train_telco.to_csv('../data/telco_churn_train.csv', index=False)
test_telco.to_csv('../data/telco_churn_test.csv', index=False)
print("Telco Churn processed and saved.")

# %% [markdown]
# ### Step 2: Heart Disease Preprocessing
# Source: UCI ML Repository (Cleveland version, ID 45)

# %%
print("Downloading Heart Disease dataset...")
heart_disease = fetch_ucirepo(id=45)
df_heart = heart_disease.data.features.copy()
df_heart['target'] = heart_disease.data.targets

# 1. Binarise target: 0 = no disease, 1-4 = disease present (so >0 means 1)
df_heart['target'] = (df_heart['target'] > 0).astype(int)

# 2. Missing values are natively NaNs from ucimlrepo when parsed, but let's ensure
# If any object type exists containing '?', replace with NaN and cast to float
df_heart = df_heart.replace('?', np.nan).astype(float)
df_heart['target'] = df_heart['target'].astype(int) # Reset target to int after float cast

# 3. Stratified 80/20 train/test split
X_heart = df_heart.drop(columns=['target'])
y_heart = df_heart['target']
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(
    X_heart, y_heart, test_size=0.2, stratify=y_heart, random_state=42
)

train_heart = pd.concat([X_train_heart, y_train_heart], axis=1)
test_heart = pd.concat([X_test_heart, y_test_heart], axis=1)

train_heart.to_csv('../data/heart_disease_train.csv', index=False)
test_heart.to_csv('../data/heart_disease_test.csv', index=False)
print("Heart Disease processed and saved.")

# %% [markdown]
# ### Step 3: Pima Indians Diabetes Preprocessing
# Source: Kaggle (uciml/pima-indians-diabetes-database)

# %%
print("Downloading Pima Indians Diabetes dataset...")
kaggle.api.dataset_download_files('uciml/pima-indians-diabetes-database', path='../data', unzip=True)
df_pima = pd.read_csv('../data/diabetes.csv')

# 1. Replace impossible zeros with NaN
cols_with_impossible_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_pima[cols_with_impossible_zeros] = df_pima[cols_with_impossible_zeros].replace(0, np.nan)

# 2. Stratified 80/20 train/test split
X_pima = df_pima.drop(columns=['Outcome'])
y_pima = df_pima['Outcome']
X_train_pima, X_test_pima, y_train_pima, y_test_pima = train_test_split(
    X_pima, y_pima, test_size=0.2, stratify=y_pima, random_state=42
)

train_pima = pd.concat([X_train_pima, y_train_pima], axis=1)
test_pima = pd.concat([X_test_pima, y_test_pima], axis=1)

train_pima.to_csv('../data/pima_diabetes_train.csv', index=False)
test_pima.to_csv('../data/pima_diabetes_test.csv', index=False)
print("Pima Indians Diabetes processed and saved.")

# %% [markdown]
# ### Step 4: Record Baseline Statistics

# %%
import json

stats = {
    "Telco_Churn": {
        "train_rows": len(train_telco),
        "test_rows": len(test_telco),
        "features": len(X_telco.columns),
        "target_balance_pct": round(y_telco.mean() * 100, 2),
        "missing_values": int(df_telco.isna().sum().sum())
    },
    "Heart_Disease": {
        "train_rows": len(train_heart),
        "test_rows": len(test_heart),
        "features": len(X_heart.columns),
        "target_balance_pct": round(y_heart.mean() * 100, 2),
        "missing_values": int(df_heart.isna().sum().sum())
    },
    "Pima_Diabetes": {
        "train_rows": len(train_pima),
        "test_rows": len(test_pima),
        "features": len(X_pima.columns),
        "target_balance_pct": round(y_pima.mean() * 100, 2),
        "missing_values": int(df_pima.isna().sum().sum())
    }
}

os.makedirs('../outputs/results', exist_ok=True)
with open('../outputs/results/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("Baseline statistics saved to outputs/results/dataset_stats.json")
