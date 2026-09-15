# Crime Prediction — Explainable AI

![Dashboard](docs/dashboard.png)

A crime-type prediction system for San Francisco incident data that pairs ensemble ML with explainable AI — every prediction ships with a SHAP/LIME explanation, not just a label, so the output is auditable rather than a black box.

## Why explainability is the point

A model that predicts a crime category without explaining why is not something a law enforcement or urban-planning stakeholder can act on or defend. This project treats SHAP and LIME as first-class outputs alongside the prediction itself — the Streamlit dashboard surfaces feature importance and per-prediction explanations, not just an accuracy score.

## What's in it

- Three classical ML algorithms (Random Forest, Gradient Boosting, Logistic Regression) plus a TensorFlow neural network, trained and compared with automatic best-model selection
- SHAP and LIME integration for both global feature importance and per-prediction explanations
- Geospatial analysis — crime hotspot mapping over the city
- An interactive Streamlit dashboard for real-time predictions, model analytics, and hotspot visualization

## Setup

```bash
python -m venv crime_env
source crime_env/bin/activate      # crime_env\Scripts\activate on Windows
pip install -r requirements.txt
python setup_and_run.py
```

Requires Python 3.11 for full compatibility (TensorFlow's Windows wheel has known issues on 3.13 — installing `tensorflow-cpu` in place of `tensorflow` resolves it if you hit DLL errors).

### Data

The San Francisco crime dataset isn't bundled in the repo (file size). Get it from [Kaggle](https://www.kaggle.com/c/sf-crime) or data.gov and place it at `data/train.csv` — required columns are `X, Y` (coordinates), `Category`, `Dates`, `PdDistrict`, `DayOfWeek`, `Resolution`, `Address`. Without a dataset present, the system generates sample data so the pipeline can be exercised end to end.

## Usage

```bash
python main.py           # train models
python setup_and_run.py  # launch the dashboard
```

## Stack

Python · TensorFlow · scikit-learn · SHAP · LIME · Streamlit
