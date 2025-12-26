# Linear Models & Decision Trees for Geospatial Classification

End-to-end implementation and evaluation of **ridge regression**, **logistic regression (SGD in PyTorch)**, and **decision trees** for classifying European cities into countries using only **longitude/latitude** features.  [oai_citation:0‡Exercise_3_2025 (3).pdf](sediment://file_00000000722871fd9941dffa9238613a)

## What’s inside
This project focuses on three main themes:
- **Closed-form Ridge Regression** (binary classification with regularization sweep)
- **Optimization from scratch**: NumPy gradient descent on a simple convex objective
- **Logistic Regression with SGD (PyTorch)** for binary + multiclass classification
- **Decision Trees** (shallow vs. deep) and their effect on bias/variance

## Results snapshot
- **Ridge Regression (binary):** best validation at **λ = 2**, with **val ≈ 0.975** and **test ≈ 0.97**.  [oai_citation:1‡Report IML3.pdf](sediment://file_00000000c33471fdba950cdfc1cef9eb)  
- **Logistic Regression (binary):** train/val/test losses converge tightly (~**0.08–0.09**) with strong generalization behavior.  [oai_citation:2‡Report IML3.pdf](sediment://file_00000000c33471fdba950cdfc1cef9eb)  
- **Multiclass Logistic Regression:** best initial learning rate achieves **test accuracy ≈ 0.842** (with LR decay).  [oai_citation:3‡Report IML3.pdf](sediment://file_00000000c33471fdba950cdfc1cef9eb)  
- **Decision Trees (multiclass):**
  - depth=2: **test ≈ 0.750** (underfits)
  - depth=10: **test ≈ 0.997** (nearly perfect separation on this dataset)  [oai_citation:4‡Report IML3.pdf](sediment://file_00000000c33471fdba950cdfc1cef9eb)  
- **Regularization (multiclass LR):** best result was **λ = 0**; larger λ values underfit and degrade accuracy.  [oai_citation:5‡Report IML3.pdf](sediment://file_00000000c33471fdba950cdfc1cef9eb)  

## Repository contents
- `ridge_experiment.py` — ridge regression (analytical solution) + λ sweep
- `gradient_descent.py` — NumPy gradient descent trajectory visualization
- `models.py` — PyTorch model(s) used for logistic regression
- `logreg_experiment.py` — binary logistic regression (SGD) experiments
- `logreg_multiclass_experiment.py` — multiclass logistic regression + LR decay schedule
- `multiclass_trees_and_ridge.py` — decision tree experiments (shallow vs deep) + comparisons
- `docs/technical_report.pdf` — full write-up and plots

## Data
This repository does **not** include dataset CSV files.
Place them in the project root (or update file paths inside the scripts):
- `train.csv`, `validation.csv`, `test.csv`
- `train_multiclass.csv`, `validation_multiclass.csv`, `test_multiclass.csv` (if used in your setup)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
