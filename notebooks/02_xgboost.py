# %% [markdown]
# # Part 3: Method 1: XGBoost (The Careful Human Approach)
# This script benchmarks XGBoost, incorporating manual feature engineering and hyperparameter tuning.

# %%
import pandas as pd
import numpy as np
import time
import json
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Dictionary to hold results
results = {}

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, 2, 3] # SMOTE handles imbalance, but keeping as search space
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% [markdown]
# ### Telco Churn (With Feature Engineering)

# %%
print("--- XGBoost on Telco Churn ---")
train_telco = pd.read_csv('../data/telco_churn_train.csv')
test_telco = pd.read_csv('../data/telco_churn_test.csv')

X_train = train_telco.drop(columns=['Churn'])
y_train = train_telco['Churn']
X_test = test_telco.drop(columns=['Churn'])
y_test = test_telco['Churn']

start_time = time.time()

# 1. Feature Engineering
def engineer_telco(df):
    df = df.copy()
    # Binary encode yes/no columns
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    
    # One-hot encode nominal categoricals
    cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
    
    # Interaction feature
    df['cost_per_month_loyalty'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # Service count feature
    service_cols = ['OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 
                    'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']
    existing_service_cols = [c for c in service_cols if c in df.columns]
    df['service_count'] = df[existing_service_cols].sum(axis=1) if existing_service_cols else 0
    
    # Log-transform TotalCharges (handle 0s)
    df['TotalCharges_log'] = np.log1p(df['TotalCharges'])
    df = df.drop(columns=['TotalCharges'])
    
    return df

X_train_eng = engineer_telco(X_train)
X_test_eng = engineer_telco(X_test)

# Align columns (just in case get_dummies produces different columns)
X_train_eng, X_test_eng = X_train_eng.align(X_test_eng, join='left', axis=1, fill_value=0)

# 2. SMOTE for class imbalance (on training only)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_eng, y_train)

# 3. Train and Tune
baseline_clf = XGBClassifier(random_state=42, eval_metric='logloss')
baseline_clf.fit(X_train_res, y_train_res)
y_pred_base = baseline_clf.predict(X_test_eng)
print(f"Untuned F1: {f1_score(y_test, y_pred_base):.4f}")

xgb_clf = XGBClassifier(random_state=42, eval_metric='logloss')
random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=20, 
                                   scoring='f1', cv=cv, random_state=42, n_jobs=-1)
random_search.fit(X_train_res, y_train_res)

best_xgb_telco = random_search.best_estimator_
y_pred_tuned = best_xgb_telco.predict(X_test_eng)
y_prob_tuned = best_xgb_telco.predict_proba(X_test_eng)[:, 1]

end_time = time.time()
telco_time = end_time - start_time

results['Telco_Churn'] = {
    'F1': f1_score(y_test, y_pred_tuned),
    'AUC': roc_auc_score(y_test, y_prob_tuned),
    'Accuracy': accuracy_score(y_test, y_pred_tuned),
    'Time_Seconds': telco_time
}
print(f"Tuned F1: {results['Telco_Churn']['F1']:.4f}")
print(f"Total Time (Eng + Tune): {telco_time:.2f}s")

# %% [markdown]
# ### Feature Engineering Impact (Experiment C)

# %%
print("Running Feature Engineering Impact Experiment (Experiment C)...")
# 1. Raw features (only one-hot encoding for XGBoost to run)
X_train_raw = pd.get_dummies(X_train, drop_first=True)
X_test_raw = pd.get_dummies(X_test, drop_first=True)
X_train_raw, X_test_raw = X_train_raw.align(X_test_raw, join='left', axis=1, fill_value=0)
xgb_raw = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train_raw, y_train)
f1_raw = f1_score(y_test, xgb_raw.predict(X_test_raw))

# 2. Fully engineered (already have best_xgb_telco results)
print(f"F1 (Raw features): {f1_raw:.4f}")
print(f"F1 (Engineered): {results['Telco_Churn']['F1']:.4f}")
results['Experiment_C_FeatEng'] = {'F1_Raw': f1_raw, 'F1_Engineered': results['Telco_Churn']['F1']}

# %% [markdown]
# ### SHAP Analysis for Telco Churn

# %%
explainer = shap.TreeExplainer(best_xgb_telco)
shap_values = explainer.shap_values(X_test_eng)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_eng, show=False)
plt.tight_layout()
plt.savefig('../outputs/charts/shap_telco_churn.png', bbox_inches='tight')
print("SHAP plot saved to outputs/charts/shap_telco_churn.png")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_5_idx = np.argsort(mean_abs_shap)[-5:][::-1]
top_5_features = X_test_eng.columns[top_5_idx].tolist()
print("Top 5 Features by SHAP:", top_5_features)

# %% [markdown]
# ### Heart Disease & Pima Diabetes (Standard Processing)

# %%
def run_xgboost_standard(dataset_name, train_path, test_path, target_col):
    print(f"--- XGBoost on {dataset_name} ---")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    start_time = time.time()
    
    # Imputation & Scaling
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Assuming all features are numeric for these two datasets
    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_proc = scaler.transform(imputer.transform(X_test))
    
    # Train and Tune
    xgb_clf = XGBClassifier(random_state=42, eval_metric='logloss')
    random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=20, 
                                       scoring='f1', cv=cv, random_state=42, n_jobs=-1)
    random_search.fit(X_train_proc, y_train)
    
    best_clf = random_search.best_estimator_
    y_pred = best_clf.predict(X_test_proc)
    y_prob = best_clf.predict_proba(X_test_proc)[:, 1]
    
    end_time = time.time()
    
    res = {
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Time_Seconds': end_time - start_time
    }
    print(f"Tuned F1: {res['F1']:.4f}, Time: {res['Time_Seconds']:.2f}s")
    return res

results['Heart_Disease'] = run_xgboost_standard('Heart Disease', '../data/heart_disease_train.csv', '../data/heart_disease_test.csv', 'target')
results['Pima_Diabetes'] = run_xgboost_standard('Pima Diabetes', '../data/pima_diabetes_train.csv', '../data/pima_diabetes_test.csv', 'Outcome')

# %% [markdown]
# ### Dirty Data Tolerance (Experiment B)

# %%
print("--- Experiment B: Dirty Data Tolerance (Pima Diabetes) ---")
train_pima = pd.read_csv('../data/pima_diabetes_train.csv')
test_pima = pd.read_csv('../data/pima_diabetes_test.csv')
X_train = train_pima.drop(columns=['Outcome'])
y_train = train_pima['Outcome']
X_test = test_pima.drop(columns=['Outcome'])
y_test = test_pima['Outcome']

# 1. XGBoost with NO imputation (XGBoost can handle NaNs, but let's measure its performance without explicit median imputation)
xgb_dirty = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train, y_train)
f1_dirty = f1_score(y_test, xgb_dirty.predict(X_test))

print(f"XGBoost (No explicit imputation, native missing handling) F1: {f1_dirty:.4f}")
print(f"XGBoost (Median Imputation + Tuned) F1: {results['Pima_Diabetes']['F1']:.4f}")
results['Experiment_B_DirtyData'] = {'XGBoost_NoImpute_F1': f1_dirty, 'XGBoost_Imputed_F1': results['Pima_Diabetes']['F1']}

# Save Results
with open('../outputs/results/xgboost_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("All XGBoost results saved.")
