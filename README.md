# NEO Hazard Classification

Machine learning project that classifies Near-Earth Objects (NEOs) as
potentially hazardous or non-hazardous based on orbital and physical
characteristics.

## Overview
- **Goal:** Predict whether an NEO is Potentially Hazardous (binary classification)
- **Data:** [NASA Near-Earth Asteroids and Close Approaches dataset](https://www.kaggle.com/datasets/darkmatternet/nasa-near-earth-asteroids-and-close-approaches) (Kaggle), sourced from NASA JPL's Center for Near-Earth Object Studies (CNEOS)
- **Business value:** Automates hazard classification to help prioritize which near-Earth objects require closer monitoring, reducing manual review time for planetary defense decisions
- **Approach:** Data cleaning, leakage-aware feature engineering, model comparison, SHAP-based explainability, and a deployed interactive demo
- **Challenge:** Highly imbalanced classes (~93.8% non-hazardous vs. ~6.2% hazardous)

## Features
- Exploratory data analysis and visualisation
- Leakage-safe preprocessing (excludes features that directly define the target by NASA's rule)
- Domain-informed feature engineering from orbital elements, validated via correlation and feature importance
- Class-imbalance handling (class weighting and SMOTE)
- Multiple classifiers compared (Logistic Regression, Random Forest, XGBoost) via stratified cross-validation
- Evaluation with precision, recall, F1-score, ROC-AUC, and confusion matrix
- **Explainable AI (SHAP)** — global feature impact and per-prediction explanations
- **Live interactive demo** deployed with Streamlit, showing predictions with confidence scores and SHAP explanations

## Tech Stack
Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · SHAP · matplotlib / seaborn · Streamlit · joblib

## Results
- **Best model:** Random Forest (selected over XGBoost and Logistic Regression via stratified 5-fold CV)
- **ROC-AUC:** 0.9675
- **Recall (Hazardous class):** 77.6% (394 / 508 hazardous objects correctly identified)
- **F1-Score (Hazardous class):** ~0.59
- Top predictive features: absolute magnitude (`H`), observation data arc, perihelion distance (`q`), and two engineered features (`orbit_spread`, `q_proximity_1au`)

## Live Demo
https://neohazardclassification.streamlit.app/

## Project Structure
neo-hazard-classification/
├── data/ # processed data, scaler, feature column list
├── notebooks/ # EDA, preprocessing, feature engineering, model selection/training/evaluation
├── models/ # saved final model
├── results/ # saved evaluation plots (ROC curve, confusion matrix, SHAP summary)
├── app.py # Streamlit deployment app
├── requirements.txt
└── README.md


## Getting Started

**Run the demo locally:**
```bash
git clone <repo-url>
cd neo-hazard-classification
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

**Explore the notebooks:**
Open any notebook in `notebooks/` in Jupyter or VS Code to walk through the full pipeline from raw data to final model.

## Author
Thisali Aturusinghe - SLIIT Kandy Uni HDIT Y2S1 - Foundations of AI - Individual Project
