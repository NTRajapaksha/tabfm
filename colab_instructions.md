# Running TabFM on Google Colab

To run TabFM on the full dataset using a free Colab T4 GPU (16GB VRAM), follow these steps.

## Step 1: Open Colab & Set GPU
1. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Go to **Runtime > Change runtime type**.
3. Select **T4 GPU** and click Save.

## Step 2: Upload Your Data
In the left sidebar of Colab, click the **Folder icon** (Files). 
Drag and drop the following files from your local `tabfm_benchmark/data` folder into the Colab file explorer:
- `telco_churn_train.csv` and `telco_churn_test.csv`
- `heart_disease_train.csv` and `heart_disease_test.csv`
- `pima_diabetes_train.csv` and `pima_diabetes_test.csv`

## Step 3: Run the Setup Cell
Copy and paste this into the first cell and run it to install the environment:

```python
!git clone https://github.com/google-research/tabfm.git
%cd tabfm
!pip install -e ".[pytorch]"
!pip install safetensors psutil pandas scikit-learn
```

## Step 3: Download Model Weights
Run this in a new cell to securely download the weights using your Hugging Face token:

```python
!pip install hf_transfer -q
import os
from google.colab import userdata
from huggingface_hub import snapshot_download

# Fetch the token securely from Colab Secrets
hf_token = userdata.get('HF_TOKEN')

snapshot_download(
    repo_id="google/tabfm-1.0.0-pytorch",
    local_dir="/content/tabfm_weights",
    token=hf_token,
    max_workers=1
)
```

## Step 4: Run the Setup Cell
Create a new cell, copy and paste the code below, and run it. This is your `04_tabfm.py` script but modified to run directly in Colab without the 200-row limit, and pointing to the Colab file paths.

```python
import pandas as pd
import numpy as np
import time
import json
import os
import psutil
import torch
from google.colab import drive
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tabfm import tabfm_v1_0_0_pytorch
from tabfm import TabFMClassifier
import math
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive to save results persistently
drive.mount('/content/drive')
output_dir = '/content/drive/MyDrive/TabFM_Benchmark'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'tabfm_results_colab.json')

print("Loading TabFM model from local weights...")
# Using bfloat16 keeps it well within the 16GB T4 GPU VRAM
model = tabfm_v1_0_0_pytorch.load(
    model_type="classification", 
    checkpoint_path="/content/tabfm_weights",
    dtype=torch.bfloat16, 
    device="cuda"
)

def run_tabfm(train_path, test_path, target_col, dataset_name, preset=None):
    print(f"\n--- TabFM ({preset or 'plain'}) on {dataset_name} ---")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Optional safety cap for T4 (Colab has 16GB, full 5000 rows might still push the limit for ensemble)
    if len(train_data) > 2500:
        print(f"  [WARNING] Subsampling {dataset_name} from {len(train_data)} to 2500 rows for memory.")
        train_data = train_data.sample(n=2500, random_state=42)
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    if preset == "ensemble":
        clf = TabFMClassifier.ensemble(model=model)
    else:
        clf = TabFMClassifier(model=model)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    
    batch_size = 64
    y_pred = []
    y_prob_list = []
    
    print(f"Predicting (Total batches: {math.ceil(len(X_test)/batch_size)})...")
    for i in range(0, len(X_test), batch_size):
        print(f"  Processing batch {i//batch_size + 1}...")
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
        auc = roc_auc_score(y_test, y_prob_list) if y_prob_list else None
    except Exception:
        auc = None
        
    end_time = time.time()
    
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    
    res = {
        'F1_Macro': f1,
        'AUC': auc,
        'Accuracy': acc,
        'Time_Seconds': end_time - start_time,
    }
    
    print(f"F1: {f1:.4f}, Time: {res['Time_Seconds']:.2f}s")
    return res

datasets = [
    ('Heart_Disease', '/content/heart_disease_train.csv', '/content/heart_disease_test.csv', 'target'),
    ('Pima_Diabetes', '/content/pima_diabetes_train.csv', '/content/pima_diabetes_test.csv', 'Outcome'),
    ('Telco_Churn', '/content/telco_churn_train.csv', '/content/telco_churn_test.csv', 'Churn')
]

# Load existing results if resuming from a crash
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        results = json.load(f)
    print("Loaded existing checkpoints from Google Drive.")
else:
    results = {}

for ds_name, train_p, test_p, target_c in datasets:
    if ds_name not in results:
        results[ds_name] = {}
        
    for mode, preset_val in [('Plain', None), ('Ensemble', 'ensemble')]:
        if mode not in results[ds_name]:
            res = run_tabfm(train_p, test_p, target_c, ds_name, preset=preset_val)
            results[ds_name][mode] = res
            
            # Checkpoint to Google Drive immediately after every run!
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"--> Checkpointed {ds_name} {mode} to Drive.")
        else:
            print(f"Skipping {ds_name} {mode} (already completed)")

print(f"\nAll TabFM results securely saved to your Google Drive at {output_file}")
print("Please download this file from your Google Drive and move it to your local outputs/results/ folder.")
```

## Step 5: Merge Results Locally
Once the script finishes on Colab:
1. In the left sidebar of Colab, open the `results` folder.
2. Download `tabfm_results_colab.json`.
3. Rename it to `tabfm_results.json` and replace the one in your local `tabfm_benchmark/outputs/results/` folder.
4. Run `05_comparison.py` on your local VS Code to generate the final matrix!
