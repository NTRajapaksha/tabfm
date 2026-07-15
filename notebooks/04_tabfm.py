# %% [markdown]
# # Part 5: Method 3: TabFM (Zero-Shot)
# This script runs the newly released Google TabFM in both plain and ensemble modes, 
# and performs the low-data regime experiment (Experiment A).

# %%
import pandas as pd
import numpy as np
import time
import json
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import math
from tabfm import tabfm_v1_0_0_pytorch # Note: Import path may need validation depending on TabFM repo structure
from tabfm import TabFMClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

results = {}
os.makedirs('../outputs/results', exist_ok=True)

# %% [markdown]
# ### Load the Model

# %%
print("Loading TabFM model in bfloat16 to save memory (may download ~7GB weights on first run)...")
# Force load in bfloat16 and explicitly onto CUDA (GPU). If it defaults to CPU, it will be extremely slow!
model = tabfm_v1_0_0_pytorch.load(model_type="classification", dtype=torch.bfloat16, device="cuda")

def run_tabfm(train_path, test_path, target_col, dataset_name, preset=None, max_train_samples=200):
    print(f"\n--- TabFM ({preset or 'plain'}) on {dataset_name} ---")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    if len(train_data) > max_train_samples:
        print(f"  Subsampling training data from {len(train_data)} to {max_train_samples} rows to prevent OOM...")
        train_data = train_data.sample(n=max_train_samples, random_state=42)
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Initialize TabFMClassifier
    if preset == "ensemble":
        clf = TabFMClassifier.ensemble(model=model)
    else:
        clf = TabFMClassifier(model=model)
    
    # Fit (prepares encoders/context, doesn't train weights)
    start_time = time.time()
    clf.fit(X_train, y_train)
    
    # Predict in small batches to prevent OOM during inference attention calculation
    batch_size = 64
    y_pred = []
    y_prob_list = []
    print(f"Predicting (Total batches: {math.ceil(len(X_test)/batch_size)})...")
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test.iloc[i:i+batch_size]
        
        # Single forward pass per batch
        try:
            proba = clf.predict_proba(X_batch)
            y_pred.extend(np.argmax(proba, axis=1))
            y_prob_list.extend(proba[:, 1])
        except Exception:
            # Fallback if proba fails
            y_pred.extend(clf.predict(X_batch))
            
    try:
        if y_prob_list:
            auc = roc_auc_score(y_test, y_prob_list)
        else:
            auc = None
    except Exception:
        auc = None
        
    end_time = time.time()
    end_mem = get_memory_usage()
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    
    res = {
        'F1_Macro': f1,
        'AUC': auc,
        'Accuracy': acc,
        'Time_Seconds': end_time - start_time
    }
    
    print(f"F1: {f1:.4f}, Time: {res['Time_Seconds']:.2f}s")
    return res

# %% [markdown]
# ### Run TabFM Plain & Ensemble on all datasets

# %%
datasets = [
    ('Heart_Disease', '../data/heart_disease_train.csv', '../data/heart_disease_test.csv', 'target'),
    ('Pima_Diabetes', '../data/pima_diabetes_train.csv', '../data/pima_diabetes_test.csv', 'Outcome'), # TabFM handles NaNs implicitly
    ('Telco_Churn', '../data/telco_churn_train.csv', '../data/telco_churn_test.csv', 'Churn')
]

for ds_name, train_p, test_p, target_c in datasets:
    results[ds_name] = {}
    
    # Plain
    res_plain = run_tabfm(train_p, test_p, target_c, ds_name, preset=None)
    results[ds_name]['Plain'] = res_plain
    
    # Ensemble (Adds cross features, SVD, blending)
    # Note: Ensemble uses much more memory. You might need to subsample if it crashes on large datasets.
    res_ensemble = run_tabfm(train_p, test_p, target_c, ds_name, preset="ensemble")
    results[ds_name]['Ensemble'] = res_ensemble

# %% [markdown]
# ### Low-Data Regime (Experiment A)
# Testing F1 vs Training Size on Telco Churn

# %%
print("\n--- Experiment A: Low-Data Regime (Telco Churn) ---")
train_telco = pd.read_csv('../data/telco_churn_train.csv')
test_telco = pd.read_csv('../data/telco_churn_test.csv')

X_test = test_telco.drop(columns=['Churn'])
y_test = test_telco['Churn']

# Using XGBoost default without hyperparameter tuning for fair out-of-the-box comparison in this loop
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

train_sizes = [100, 200, 500, 1000, 2000, 5000]
low_data_results = {'Train_Size': [], 'TabFM_F1': [], 'XGBoost_F1': []}

for size in train_sizes:
    if size > len(train_telco):
        continue
    
    # Subsample training data
    train_subset = train_telco.sample(n=size, random_state=42)
    X_train_sub = train_subset.drop(columns=['Churn'])
    y_train_sub = train_subset['Churn']
    
    print(f"Testing Train Size: {size}")
    
    # TabFM
    clf_tabfm = TabFMClassifier(model=model)
    clf_tabfm.fit(X_train_sub, y_train_sub)
    
    # Batch predict for low-data experiment too
    y_pred_tabfm = []
    for i in range(0, len(X_test), 64):
        y_pred_tabfm.extend(clf_tabfm.predict(X_test.iloc[i:i+64]))
        
    f1_tabfm = f1_score(y_test, y_pred_tabfm, average='macro')
    
    # XGBoost (Requires numeric encoding)
    X_train_xgb = pd.get_dummies(X_train_sub, drop_first=True)
    X_test_xgb = pd.get_dummies(X_test, drop_first=True)
    X_train_xgb, X_test_xgb = X_train_xgb.align(X_test_xgb, join='left', axis=1, fill_value=0)
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb.fit(X_train_xgb, y_train_sub)
    y_pred_xgb = xgb.predict(X_test_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
    
    low_data_results['Train_Size'].append(size)
    low_data_results['TabFM_F1'].append(f1_tabfm)
    low_data_results['XGBoost_F1'].append(f1_xgb)
    
    print(f"  TabFM F1: {f1_tabfm:.4f} | XGBoost F1: {f1_xgb:.4f}")

results['Experiment_A_LowData'] = low_data_results

# Save Results
with open('../outputs/results/tabfm_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("\nAll TabFM results saved.")
