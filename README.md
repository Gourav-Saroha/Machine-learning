# Ensemble Learning Studio

Ensemble Learning Studio is a Flask-based Machine Learning dashboard built from an existing Jupyter Notebook on ensemble learning. The application preserves the notebook's Iris-based workflow and turns it into a professional, interactive, multi-page web experience with live Plotly charts and reusable Flask APIs.

## Project Overview

This project covers:

- Ensemble learning intuition
- Bootstrap sampling with replacement
- Decision Tree classification
- Bagging with `BaggingClassifier`
- Random Forest with `RandomForestClassifier`
- OOB score and OOB error analysis
- Gini / MDI feature importance
- Permutation importance
- Bias-variance interpretation

The application reuses the notebook's core implementation choices wherever possible:

- Iris dataset from `sklearn.datasets.load_iris`
- Shared stratified split using `train_test_split(test_size=0.2, random_state=42, stratify=y)`
- Grid search for Decision Tree, Bagging, and Random Forest
- OOB scoring for ensemble models where applicable
- Plotly-based live visual updates through Flask API routes

## Features

- Modern sidebar dashboard layout with responsive pages
- Live model retraining for Decision Tree, Bagging, and Random Forest
- Separate hyperparameter tuning actions using the notebook's original grids
- Interactive bootstrap sampling visualizations
- Real confusion matrices, performance charts, and prediction distributions
- Model comparison page using the same train/test split for all models
- Toggle between Gini importance and permutation importance
- OOB score trend chart computed from real model fits
- Bias-variance page combining conceptual explanation with observed train/test behavior

## Tech Stack

- Backend: Python, Flask
- ML: scikit-learn, pandas, numpy
- Visualization: Plotly
- Frontend: HTML, CSS, JavaScript, Bootstrap 5

## Architecture

```text
project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- models/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- decision_tree.py
|   |-- bagging.py
|   |-- random_forest.py
|   `-- evaluation.py
|-- services/
|   |-- __init__.py
|   |-- bootstrap_service.py
|   |-- model_service.py
|   |-- feature_importance.py
|   `-- oob_service.py
|-- templates/
|   |-- base.html
|   |-- overview.html
|   |-- bootstrap.html
|   |-- decision_tree.html
|   |-- bagging.html
|   |-- random_forest.html
|   |-- comparison.html
|   |-- importance.html
|   |-- oob.html
|   `-- bias_variance.html
`-- static/
	|-- css/
	|   `-- style.css
	`-- js/
		`-- dashboard.js
```

## How Ensemble Learning Works

Ensemble methods combine multiple learners to improve generalization.

- A single Decision Tree can achieve low bias but high variance.
- Bagging reduces variance by training many trees on bootstrap samples and aggregating predictions.
- Random Forest extends Bagging by adding random feature selection, which reduces tree correlation further.
- OOB samples provide an internal estimate of generalization performance without requiring a separate validation set for every tree.

## Notebook To Flask Conversion

The notebook was converted into Flask architecture by separating concerns:

- Notebook dataset loading became `models/data_loader.py`
- Notebook model definitions and parameter grids became `models/decision_tree.py`, `models/bagging.py`, and `models/random_forest.py`
- Evaluation logic moved into `models/evaluation.py`
- Interactive training, caching, and Plotly chart generation moved into the `services/` layer
- Flask routes in `app.py` now render pages and expose JSON APIs for live updates
- The frontend in `templates/` and `static/` consumes those APIs and redraws charts without page reloads

This keeps the Machine Learning logic reusable and avoids duplicating model code across routes.

## API Endpoints

- `GET /`
- `GET /bootstrap`
- `GET /decision-tree`
- `GET /bagging`
- `GET /random-forest`
- `GET /comparison`
- `GET /feature-importance`
- `GET /oob`
- `GET /bias-variance`

- `POST /api/bootstrap`
- `POST /api/decision-tree/train`
- `POST /api/decision-tree/tune`
- `POST /api/bagging/train`
- `POST /api/bagging/tune`
- `POST /api/random-forest/train`
- `POST /api/random-forest/tune`
- `GET /api/model-comparison`
- `GET /api/feature-importance`
- `GET /api/oob`

## How To Run Locally

```powershell
cd C:\Users\gourav.lnu\Downloads\ensemble-learning-flask-ui
"C:/Program Files/Python314/python.exe" -m pip install -r requirements.txt
"C:/Program Files/Python314/python.exe" app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Screenshots

- Overview page screenshot placeholder
- Bootstrap Sampling page screenshot placeholder
- Random Forest page screenshot placeholder

## Future Improvements

- Add AdaBoost, Gradient Boosting, XGBoost, LightGBM, Voting, and Stacking modules later
- Support user-uploaded tabular datasets
- Add model persistence and experiment history
- Add SHAP or partial dependence visualizations
- Add downloadable model reports and chart exports
