# %% [markdown]
# # Part 4: Method 2: AutoGluon (AutoML)
# This script runs AutoGluon on the raw/minimally-processed datasets 
# across three time budgets: 60s, 10m, and 1h.

# %%
import pandas as pd
import time
import json
import os
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

results = {}
os.makedirs('../outputs/results', exist_ok=True)

RUN_FULL = False # Set to True to run the 1-hour budgets

# Function to evaluate AutoGluon for a specific time budget
def run_autogluon_budget(train_path, test_path, target_col, time_limit, dataset_name):
    print(f"\n--- AutoGluon on {dataset_name} ({time_limit}s limit) ---")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Define save path for the model
    save_path = f'../outputs/agModels_{dataset_name.replace(" ", "_")}_{time_limit}s'
    
    start_time = time.time()
    
    # Train the predictor
    predictor = TabularPredictor(label=target_col, eval_metric='f1_macro', path=save_path).fit(
        train_data, time_limit=time_limit, presets='best_quality'
    )
    
    end_time = time.time()
    
    # Predictions
    y_test = test_data[target_col]
    y_pred = predictor.predict(test_data)
    
    try:
        y_prob = predictor.predict_proba(test_data).iloc[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None # Some configurations might not output probas
        
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    
    # Leaderboard
    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.iloc[0]['model']
    
    # Disk space
    def get_size(start_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
        
    disk_space_mb = get_size(save_path) / (1024 * 1024)
    
    res = {
        'F1_Macro': f1,
        'AUC': auc,
        'Accuracy': acc,
        'Time_Seconds': end_time - start_time,
        'Best_Model': best_model,
        'Models_Evaluated': len(leaderboard),
        'Disk_Space_MB': round(disk_space_mb, 2),
        'Used_Stacking_Ensembling': any('stack' in m.lower() or 'ensemble' in m.lower() for m in leaderboard['model'])
    }
    
    print(f"F1: {f1:.4f}, Time: {res['Time_Seconds']:.2f}s, Best: {best_model}")
    return res

# %% [markdown]
# ### Run budgets for all datasets

# %%
datasets = [
    ('Heart_Disease', '../data/heart_disease_train.csv', '../data/heart_disease_test.csv', 'target'),
    ('Pima_Diabetes', '../data/pima_diabetes_train.csv', '../data/pima_diabetes_test.csv', 'Outcome'),
    ('Telco_Churn', '../data/telco_churn_train.csv', '../data/telco_churn_test.csv', 'Churn')
]

time_budgets = [60, 600, 3600] # 60s, 10m, 1h

for ds_name, train_p, test_p, target_c in datasets:
    results[ds_name] = {}
    for budget in time_budgets:
        if budget == 3600 and not RUN_FULL:
            print(f"Skipping 3600s budget for {ds_name} (RUN_FULL=False)")
            continue
        res = run_autogluon_budget(train_p, test_p, target_c, budget, ds_name)
        results[ds_name][f'{budget}s'] = res

# Save Results
with open('../outputs/results/autogluon_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("\nAll AutoGluon results saved.")

# %%
