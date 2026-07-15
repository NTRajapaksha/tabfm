<div align="center">
  <h1>🌲 XGBoost vs. 🤖 TabFM</h1>
  <h3><i>Benchmarking Google's Zero-Shot Tabular Foundation Model</i></h3>
  <br>
  
  [![Read the Article](https://img.shields.io/badge/📖_Read_The_Full_Article-Medium-black?style=for-the-badge)](https://medium.com/@ntratofficial/is-the-reign-of-xgboost-over-i-benchmarked-googles-new-tabfm-and-the-results-are-wild-d1ecd0a6d9f3)
</div>

---

## 🚀 The Experiment

For years, XGBoost and AutoML frameworks have ruled structured data. But Google Research recently released **TabFM**, claiming state-of-the-art performance using a massive pre-trained Transformer—with **zero hyperparameter tuning** and **zero training loops**.

We decided to put that claim to the test. This repository contains a rigorous, head-to-head benchmark pitting **TabFM** against **XGBoost** and **AutoGluon** across three distinct tabular challenges:

1. 🫀 **Heart Disease** *(Small & Clean - Tests pure predictive power)*
2. 🩸 **Pima Diabetes** *(Hidden Missing Values - Tests resilience to dirty data)*
3. 📉 **Telco Churn** *(Large & Complex - Tests scaling and feature engineering)*

---

## 🏆 TL;DR The Results

| Dataset | XGBoost (Tuned) | AutoGluon (10m) | TabFM (Zero-Shot) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Heart Disease (F1)** | 0.8710 | 0.8195 | **0.8688** | 🤝 Tie (XGB/TabFM) |
| **Pima Diabetes (F1)** | 0.6610 | 0.7187 | **0.7097** | 🤖 TabFM / AutoGluon |
| **Telco Churn (F1)** | 0.5686 | **0.7382** | 0.7285 | 🏅 AutoGluon |

*💡 **The Catch:** While TabFM achieved phenomenal zero-shot accuracy and handled dirty data flawlessly, its Transformer architecture relies on quadratic attention. Predicting 2,500 rows took **70+ minutes on a T4 GPU**, compared to AutoGluon's **10 minutes on a CPU**.*

---

## 📂 Repository Structure

```text
├── data/                  # 📊 Raw CSV train/test splits for the 3 datasets
├── notebooks/             # 🧠 The Python benchmark scripts
│   ├── 01_data_prep.py    # Downloads and prepares the datasets
│   ├── 02_xgboost.py      # XGBoost baselines (raw, tuned, engineered)
│   ├── 03_autogluon.py    # AutoGluon AutoML baselines (60s, 600s budgets)
│   ├── 04_tabfm.py        # TabFM local inference script
│   └── 05_comparison.py   # Aggregates results and generates PNG charts
├── colab_instructions.md  # ☁️ How to run the heavy TabFM model on Google Colab
├── outputs/               # 📈 Generated matrices, JSON results, and visualizations
└── requirements.txt       # 📦 Python dependencies
```

---

## 🛠️ How to Reproduce the Benchmark

### 1. Local Environment Setup
Clone this repository and install the dependencies (Python 3.10+):
```bash
git clone https://github.com/yourusername/tabfm-benchmark.git
cd tabfm-benchmark
pip install -r requirements.txt
```
*(Pro-tip: To get the full AutoGluon ensemble, also run `pip install lightgbm catboost fastai`)*

### 2. Run the Tree Baselines (Local CPU)
Run the data preparation, XGBoost, and AutoGluon scripts sequentially:
```bash
cd notebooks
python 01_data_prep.py
python 02_xgboost.py
python 03_autogluon.py
```

### 3. Run TabFM (Google Colab GPU)
TabFM requires significant VRAM to hold the dataset in context. 
👉 **Please refer to the `colab_instructions.md` file** in this repo for a simple, copy-paste guide on how to run this step for free using a Colab T4 GPU. 

### 4. Generate the Visualizations
Once you've run the scripts and placed your TabFM results in the `outputs/results/` folder, run the final script to generate the comparison matrices and charts:
```bash
python 05_comparison.py
```

---

## 📄 License
The code in this repository is MIT Licensed. TabFM model weights fall under the Google Research TabFM Non-Commercial License v1.0.
