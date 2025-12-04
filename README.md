# Clinflow API
## Description
Clinflow API is a containerised, tested Python service that ingests clinical tabular data, trains a reproducible baseline risk model, and exposes predictions via a versioned FastAPI endpoint.

## Current functionality:
- Fetches/downloads UCI heart disease dataset
- Built in functions for:
    - Loading raw data
    - Cleaning raw data, and validating this cleaned data
    - Conversion to an sqlite database
    - Querying for sqlite database with flexible presets
    - Logging
    - Other utilities
and a pipeline for converting raw data to cleaned data in csv/sql form. Uses pandas (mostly) and sqlite for data handling.
- Tests for the cleaning, configuration, and data loading functions

This project is still a work in progress. Currently about 15 days/halfway into the project.

## Setup and Installation
- Environment/dependencies are configured via pyproject.toml. To set up:
  ```
  git clone https://github.com/Callum6933/clinflow-api.git
  cd clinflow-api
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -e .
  ```

## Running the pipeline
After cloning and activating venv, to run the pipeline:
```
python -m clinflow.data.download_data
python -m clinflow.pipeline
```

This will download the UCI heart disease dataset and run the full pipeline. This will create a `clean.csv` file in `data/processed/`, and a `clinflow.db` file in `data/`

## Testing
Pytest is used for this project. To run tests, run `pytest` from the root or run a specific test.