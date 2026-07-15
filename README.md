# TabFM vs The World: A Zero-Shot Tabular Benchmark

This repository contains the full benchmark code for evaluating Google Research's **TabFM (Tabular Foundation Model)** against industry-standard tree-based models: **XGBoost** and **AutoGluon**. 

The goal of this benchmark is to test TabFM's zero-shot, in-context learning capabilities on tabular data across three distinct scenarios:
1. **Heart Disease**: A small, clean dataset (Predictive Power).
2. **Pima Diabetes**: A dataset with hidden missing values (Dirty Data Resilience).
3. **Telco Churn**: A larger dataset requiring feature engineering (Scaling & Complexity).

## Repository Structure

```
├── data/                  # Raw CSV files for the 3 datasets (train/test splits)
├── notebooks/             # Python scripts to run the benchmark
│   ├── 01_data_prep.py    # Downloads and splits the datasets
│   ├── 02_xgboost.py      # XGBoost baselines (raw, tuned, engineered)
│   ├── 03_autogluon.py    # AutoGluon AutoML baselines (60s, 600s budgets)
│   ├── 04_tabfm.py        # TabFM local inference script
│   └── 05_comparison.py   # Aggregates results into matrices and generates charts
├── outputs/               # Auto-generated folder for saved models, results, and charts
│   ├── charts/            # PNG visualizations
│   └── results/           # JSON/CSV comparison matrices
└── requirements.txt       # Python dependencies
```

## How to Reproduce the Results

### 1. Local Environment Setup
Clone this repository and install the required dependencies (Python 3.10+ recommended):
```bash
git clone https://github.com/yourusername/tabfm-benchmark.git
cd tabfm-benchmark
pip install -r requirements.txt
```
*(Note: To get the full AutoGluon ensemble, ensure you also run `pip install lightgbm catboost fastai`)*

### 2. Run the Baselines (Local)
Navigate to the `notebooks/` directory and run the data preparation, XGBoost, and AutoGluon scripts:
```bash
cd notebooks
python 01_data_prep.py
python 02_xgboost.py
python 03_autogluon.py
```

### 3. Run TabFM (Google Colab)
Because TabFM uses a Transformer architecture and holds the entire training dataset in memory as context, its VRAM requirements scale quadratically. It is highly recommended to run TabFM on a GPU (like a free Colab T4).
1. Open Google Colab and select a T4 GPU runtime.
2. Upload the `data/` folder CSVs to the Colab environment.
3. Follow the instructions in `colab_instructions.md` (or copy the contents of `04_tabfm.py`) to run the inference.
4. Download the resulting `tabfm_results_colab.json` to your local `outputs/results/` folder and rename it to `tabfm_results.json`.

### 4. Generate Final Comparisons
Once all JSON result files are in the `outputs/results/` directory, run the comparison script to build the CSV matrices and PNG charts:
```bash
python 05_comparison.py
```

## Results Summary
The benchmark revealed that TabFM achieves state-of-the-art performance, matching or beating 10-minute AutoGluon ensembles instantly in a zero-shot forward pass. However, its compute time scales quadratically, making it significantly slower to infer than tree-based models on datasets with >2,000 rows.

For a full breakdown of the results, read the accompanying article.

## License
Code in this repository is MIT Licensed. TabFM model weights fall under the Google Research TabFM Non-Commercial License v1.0.
