# Predictive Maintenance AI Platform

An interactive Streamlit web app that predicts industrial machine failures and remaining useful life from sensor data.

**Live demo:** _add your Streamlit link here_

## Features
- Failure classification (`Failure_Within_7_Days`) with Random Forest, XGBoost or LightGBM
- Remaining useful life regression (`Remaining_Useful_Life_days`)
- **Live prediction**: enter one machine's sensor readings and get a risk score and estimated remaining life
- Adjustable decision threshold, ROC and Precision-Recall curves
- Anomaly detection (Isolation Forest), K-Means machine segmentation, PCA
- Explainability with permutation importance and SHAP
- Upload your own CSV / Excel file, export predictions

## Results (held-out test set, ~100k-row training sample)
| Task | Result |
|---|---|
| Failure within 7 days | ROC AUC ≈ 0.98, recall ≈ 0.92, precision ≈ 0.55 (failures are only ~6% of the data) |
| Remaining useful life | R² ≈ 0.97, MAE ≈ 39 days |

## A note on data leakage
In this dataset `Remaining_Useful_Life_days` directly reveals the failure label (and vice versa).
Keeping either one as an input feature gives a fake 100% accuracy, so both targets are always
excluded from the features, along with `Machine_ID`.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```
On Windows you can also double-click `run_app.bat`.

## Dataset
`factory_sensor_simulator_2040.csv` in this repo is a **20,000-row stratified sample** of the full
500,000-row simulated dataset, so the app runs quickly. Use the **Load** button in the sidebar, or
upload your own file.

_Educational decision-support prototype, not a certified safety system._
