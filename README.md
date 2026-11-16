AG-XGBoost

Location-adaptive bandwidth geographically weighted XGBoost. The bandwidth of each location is selected by minimising its leave-one-out error; all code is in agxgboost.py.

Install

    pip install -r requirements.txt

Quick start (built-in synthetic data)

    python agxgboost.py --demo --fast --out ./demo_out

Outputs

- metrics.json — run configuration and leave-one-out R², MAE, RMSE.
- per_location.csv — coordinates, selected bandwidth, observed and predicted values, squared error.
- shap_summary.csv — mean absolute SHAP value per feature (requires shap).

License

MIT, see LICENSE.
