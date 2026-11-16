AG-XGBoost

Location-adaptive bandwidth geographically weighted XGBoost (Hu et al., 2026, Computers & Geosciences). The bandwidth of each location is selected by minimising its leave-one-out error; all code is in agxgboost.py.

Install

    pip install -r requirements.txt

Quick start (built-in synthetic data)

    python agxgboost.py --demo --fast --out ./demo_out

Your own data

Input is a CSV: feature columns, a target column, and two projected coordinate columns (metres).

    python agxgboost.py --data data.csv --x f1 f2 f3 --y target --coords x y --out ./results

Main options

  Option                       	Default        	Meaning                                 
  --method {agxgb,gxgb}        	agxgb          	location-adaptive bandwidth / one bandwidth for all locations
  --search {coarse_golden,step}	coarse_golden  	two-stage search / stepwise search      
  --no-borrow                  	off            	one global hyper-parameter configuration instead of spatial borrowing
  --fast                       	off            	small hyper-parameter grid (quick runs) 
  --no-shap                    	off            	skip SHAP attribution                   
  --out                        	./agxgboost_out	output directory                        

Full list: python agxgboost.py -h.

Outputs

- metrics.json — run configuration and leave-one-out R², MAE, RMSE.
- per_location.csv — coordinates, selected bandwidth, observed and predicted values, squared error.
- shap_summary.csv — mean absolute SHAP value per feature (requires shap).

Data

Sources of the four data sets used in the paper: see docs/DATA.md.

License

MIT, see LICENSE.
