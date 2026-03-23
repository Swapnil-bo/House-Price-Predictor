# 🏠 House Price Predictor: A Production-Ready ML Pipeline
![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

An end-to-end Machine Learning project predicting property values using the Ames Housing Dataset. This project emphasizes **production-grade engineering practices**, including Scikit-Learn pipelines, rigorous multicollinearity checks (VIF), residual diagnostics, and a dynamic Streamlit web interface with Explainable AI (XAI) features.

---

## 🚀 Live Demo
👉 **[house-price-predictor-h4xszy5zhgrt42wjtcwkhb.streamlit.app](https://house-price-predictor-h4xszy5zhgrt42wjtcwkhb.streamlit.app/)**

---

## ✨ Key Features & Methodology
This isn't just a simple `model.fit()` script. The architecture is broken into three robust phases:

* **Phase 1: Advanced EDA & Preprocessing (`eda.py`)**
    * Automated high-missingness dropping and robust imputation (median/mode).
    * Target log-transformation (`log1p`) to correct right-skewed pricing distributions.
    * Iterative Variance Inflation Factor (VIF) pruning to eliminate multicollinearity.
* **Phase 2: Pipeline Engineering (`modeling.py`)**
    * Implementation of `sklearn.pipeline.Pipeline` to guarantee **zero data leakage** during Cross-Validation.
    * Evaluation of Linear Regression (OLS) vs. Polynomial Regression (Degree 2).
    * 5-Fold Cross-Validation logging both $R^2$ and RMSE.
* **Phase 3: Diagnostics & Evaluation (`evaluation.py`)**
    * Transformation of predictions back to original USD scale (`expm1`) for business interpretability.
    * LOWESS-smoothed residual plots to verify homoscedasticity and check for systematic bias.
    * Standardized coefficient extraction for feature importance.
* **Web App: Interactive XAI Dashboard (`app.py`)**
    * Dynamic slider ranges calibrated automatically to training data constraints.
    * Real-time confidence intervals (±1 RMSE approximation).
    * Live "Feature Impact" bar charts explaining exactly *why* a house is priced the way it is.

## 📂 Project Structure
```text
house-price-predictor/
├── .gitignore                  # Keeps venv and large files out of version control
├── train.csv                   # Raw Ames Housing dataset
├── app.py                      # Streamlit web application
├── eda.py                      # Phase 1: Cleaning & Selection
├── modeling.py                 # Phase 2: Pipeline Training
├── evaluation.py               # Phase 3: Metrics & Diagnostics
├── linear_pipeline.pkl         # Serialized baseline model
├── poly_pipeline.pkl           # Serialized complex model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
