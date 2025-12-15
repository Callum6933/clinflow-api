# Clinflow API
## Description
A tested Python pipeline that ingests clinical tabular data, cleans it, trains a reproducible baseline risk model, and evaluates predictions on heart disease.

## Current functionality:
- Fetches/downloads UCI heart disease dataset
- Data pipeline with built-in functions for:
    - Loading raw data
    - Cleaning raw data and validating cleaned data
    - Conversion to an sqlite database
    - Querying sqlite database with flexible presets
    - Configuration management (YAML)
    - Logging
    - Other utilities
- Model training pipeline with:
    - Scikit-learn preprocessing and logistic regression
    - Train/test splitting with configurable parameters
    - Model evaluation (classification metrics, confusion matrix)
    - Model persistence (save/load with joblib)
    - CLI for full train → evaluate → save workflow
- Tests for data loading, cleaning, configuration, and model training

Uses pandas and sqlite for data handling, scikit-learn for modeling.

**Project status (v0.2.0)**: Complete data and modelling pipeline. API/Docker deployment intentionally scoped out to focus on computational foundations.

## Setup and Installation
- Environment/dependencies are configured via pyproject.toml. To set up:
  ```
  git clone https://github.com/Callum6933/clinflow-api.git
  cd clinflow-api
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -e .
  ```

## Running the Data Pipeline
After cloning and activating venv, to run the data pipeline:
```
python -m clinflow.data.download_data
python -m clinflow.pipeline
```

This will download the UCI heart disease dataset and run the full pipeline. This will create a `clean.csv` file in `data/processed/`, and a `clinflow.db` file in `data/`

## Training the Model
To train the model and generate evaluation metrics:
```
# Train using the processed CSV file
python -m clinflow.models.train_model_cli --csv data/processed/clean.csv

# Or train using data from the database
python -m clinflow.models.train_model_cli --from-db --query all
```

This will train a logistic regression model, save it to `models/heart_disease_model.pkl`, and write evaluation metrics to `results/metrics.json`.

## Testing
Pytest is used for this project. To run tests, run `pytest` from the root or run a specific test.