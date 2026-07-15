# %% [markdown]
# # Part 7: Build the Comparison Matrix & Visualize
# This script aggregates the results from all methods on the Telco Churn dataset, 
# formats them into a comparison matrix, and generates the required visualisations.

# %%
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('../outputs/charts', exist_ok=True)

# %% [markdown]
# ### Load Results

# %%
def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

xgb_res = load_json('../outputs/results/xgboost_results.json')
ag_res = load_json('../outputs/results/autogluon_results.json')
tabfm_res = load_json('../outputs/results/tabfm_results.json')
llm_res = load_json('../outputs/results/llm_results.json')

# %% [markdown]
# ### Build the Comparison Matrix (Telco Churn)

# %%
datasets = ['Telco_Churn', 'Heart_Disease', 'Pima_Diabetes']

for dataset in datasets:
    matrix_data = []

    # XGBoost
    if dataset in xgb_res:
        matrix_data.append({
            'Method': 'XGBoost (tuned)',
            'F1 (test)': f"{xgb_res[dataset]['F1']:.4f}",
            'AUC': f"{xgb_res[dataset]['AUC']:.4f}",
            'Time to first prediction': f"{xgb_res[dataset]['Time_Seconds']/3600:.2f} hours (tuning)",
            'Setup effort': 'High - manual feature eng' if dataset == 'Telco_Churn' else 'Low - standard scaling',
            'Memory': 'Low',
            'Cost': 'Free',
            'Explainability': 'SHAP - excellent'
        })

    # AutoGluon
    if dataset in ag_res:
        for budget in [60, 600, 3600]:
            b_key = f"{budget}s"
            if b_key in ag_res[dataset]:
                matrix_data.append({
                    'Method': f"AutoGluon {b_key}",
                    'F1 (test)': f"{ag_res[dataset][b_key]['F1_Macro']:.4f}",
                    'AUC': f"{ag_res[dataset][b_key]['AUC']:.4f}" if ag_res[dataset][b_key]['AUC'] else "N/A",
                    'Time to first prediction': f"{ag_res[dataset][b_key]['Time_Seconds']:.0f}s",
                    'Setup effort': 'Low - pass raw data',
                    'Memory': f"{ag_res[dataset][b_key]['Disk_Space_MB']}MB (Disk)",
                    'Cost': 'Free',
                    'Explainability': 'None for ensemble'
                })

    if dataset in tabfm_res:
        for mode in ['Plain', 'Ensemble']:
            if mode in tabfm_res[dataset]:
                ram_val = tabfm_res[dataset][mode].get('Peak_RAM_MB_Delta', 'N/A')
                matrix_data.append({
                    'Method': f"TabFM ({mode})",
                    'F1 (test)': f"{tabfm_res[dataset][mode]['F1_Macro']:.4f}",
                    'AUC': f"{tabfm_res[dataset][mode]['AUC']:.4f}" if tabfm_res[dataset][mode]['AUC'] else "N/A",
                    'Time to first prediction': f"{tabfm_res[dataset][mode]['Time_Seconds']:.2f}s",
                    'Setup effort': 'Very low',
                    'Memory': f"{ram_val}MB (RAM)" if ram_val != 'N/A' else "16GB (GPU VRAM)",
                    'Cost': 'Free (non-commercial)',
                    'Explainability': 'None'
                })

    # LLM (Only ran for Telco Churn in 05 script)
    if dataset == 'Telco_Churn':
        for mode in ['Zero_Shot', 'Few_Shot']:
            if mode in llm_res:
                matrix_data.append({
                    'Method': f"Gemini Flash ({mode.replace('_', '-')})",
                    'F1 (test)': f"{llm_res[mode]['F1_Macro']:.4f}",
                    'AUC': "N/A",
                    'Time to first prediction': f"~{llm_res[mode]['Avg_Latency_Per_Row']:.2f}s per row",
                    'Setup effort': 'Low - prompting',
                    'Memory': 'None (API)',
                    'Cost': 'Free tier',
                    'Explainability': 'Can ask for explanation'
                })

    if matrix_data:
        df_matrix = pd.DataFrame(matrix_data)
        print(f"=== Comparison Matrix ({dataset}) ===")
        print(df_matrix.to_markdown(index=False))
        print("\n")
        df_matrix.to_csv(f'../outputs/results/comparison_matrix_{dataset.lower()}.csv', index=False)

# %% [markdown]
# ### Visualise Low-Data Regime (Experiment A)

# %%
if 'Experiment_A_LowData' in tabfm_res:
    low_data = tabfm_res['Experiment_A_LowData']
    
    plt.figure(figsize=(10, 6))
    plt.plot(low_data['Train_Size'], low_data['TabFM_F1'], marker='o', label='TabFM (Zero-Shot fit)', linewidth=2)
    plt.plot(low_data['Train_Size'], low_data['XGBoost_F1'], marker='s', label='XGBoost (Untuned baseline)', linewidth=2)
    
    plt.title('F1 Score vs Training Set Size (Telco Churn)', fontsize=14)
    plt.xlabel('Number of Training Rows', fontsize=12)
    plt.ylabel('Macro F1 Score', fontsize=12)
    plt.xscale('log')
    plt.xticks(low_data['Train_Size'], low_data['Train_Size'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../outputs/charts/low_data_regime.png', dpi=300)
    print("\nLow data regime chart saved to outputs/charts/low_data_regime.png")

# %% [markdown]
# ### Visualise Dirty Data & Feature Eng (Experiments B & C)

# %%
if 'Experiment_B_DirtyData' in xgb_res and 'Pima_Diabetes' in tabfm_res:
    plt.figure(figsize=(8, 5))
    methods = ['XGBoost (No Impute)', 'XGBoost (Imputed)', 'TabFM (Plain, Raw NaN)']
    f1_scores = [
        xgb_res['Experiment_B_DirtyData']['XGBoost_NoImpute_F1'],
        xgb_res['Experiment_B_DirtyData']['XGBoost_Imputed_F1'],
        tabfm_res['Pima_Diabetes'].get('Plain', {}).get('F1_Macro', 0)
    ]
    
    sns.barplot(x=methods, y=f1_scores, palette='viridis')
    plt.title('Experiment B: Dirty Data Tolerance (Pima Diabetes)')
    plt.ylabel('Macro F1 Score')
    plt.ylim(0, 1.0)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
        
    plt.tight_layout()
    plt.savefig('../outputs/charts/dirty_data_exp_b.png', dpi=300)
    print("\nExperiment B chart saved to outputs/charts/dirty_data_exp_b.png")

if 'Experiment_C_FeatEng' in xgb_res and 'Telco_Churn' in tabfm_res:
    plt.figure(figsize=(8, 5))
    methods = ['XGBoost (Raw)', 'XGBoost (Engineered)', 'TabFM (Plain, No Eng)']
    f1_scores = [
        xgb_res['Experiment_C_FeatEng']['F1_Raw'],
        xgb_res['Experiment_C_FeatEng']['F1_Engineered'],
        tabfm_res['Telco_Churn'].get('Plain', {}).get('F1_Macro', 0)
    ]
    
    sns.barplot(x=methods, y=f1_scores, palette='magma')
    plt.title('Experiment C: Feature Engineering Impact (Telco Churn)')
    plt.ylabel('Macro F1 Score')
    plt.ylim(0, 1.0)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
        
    plt.tight_layout()
    plt.savefig('../outputs/charts/feat_eng_exp_c.png', dpi=300)
    print("Experiment C chart saved to outputs/charts/feat_eng_exp_c.png")

print("\nComparison and visualization complete.")

# %%
