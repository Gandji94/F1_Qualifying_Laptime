# F1 Qualifying Times Project

## About the project
The idea behind the project is to predict the F1 qualifying times. In the project different machine learning models will be trained and evaluated, to be more specific, it compares boosting models (Light GBM, XGBoost and HistGradientBoosting) and linear models (Lasso, Ridge and Elastic Net), boosting models for prediction accuracy and linear modles for extrapolation.
These models are incorporated into a pipeline which transfrom values and select the ones with the most impact for prediction.
The output files will be processed via dbt and saved in a duckdb database.

## How to use this README

This guide describes the files and commands currently in this project. Start with setup, check the input data, and then run the workflow in order. Docker and dbt are optional routes described separately below.

Commands in `powershell` blocks are entered in Windows PowerShell. Run them from the project root: the folder containing `pyproject.toml` and `compose.yml`. Paths elsewhere in this guide are relative to that folder.

The instructions were checked against the source files; the training pipeline and tests were not executed when this README was written. See **Known limitations and troubleshooting** before starting a long run.

## Project structure

| File or folder | Purpose |
| --- | --- |
| `src/` | Python scripts for cleaning, training, evaluation, prediction, and analysis. |
| `src/main.py` | Command-line entry point that runs individual steps or a combined workflow. |
| `src/py_def_class.py` | Shared functions and custom preprocessing classes. |
| `data/raw/` | Input circuit, calendar, session, and weather data. |
| `data/processed/` | Cleaned and combined datasets. |
| `data/model_data/` | Training, testing, and final fitting datasets. |
| `data/predictions/` | Exported qualifying predictions. |
| `data/data_drift/` | Reference and newer datasets used for drift analysis. |
| `model/` | Saved models, parameters, evaluation results, offsets, and analysis outputs. |
| `tests/` | Automated data, preprocessing, training, and output checks. |
| `notebook/` | Jupyter notebooks for exploration and development. |
| `f1_qualifying_dbt/` | dbt project that loads CSV seeds and builds DuckDB views and tables. |
| `pyproject.toml` | Python version, project dependencies, and test configuration. |
| `requirment.txt` | Additional dependency list; the filename is spelled this way in the project. |
| `Dockerfile` and `compose.yml` | Container setup for the Python app and dbt. |

## 1. Set up Python

Use **Python 3.11**. The project declares `>=3.11,<3.12`, so a different minor version is outside its supported configuration.

Open PowerShell and move into the project:

```powershell
Set-Location 'C:\Users\gandj\Documents\F1\Manual'
py -3.11 --version
```

Create a virtual environment. This is a separate location for this project's Python packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`-e` installs the project so edits to the source are picked up without reinstalling it. `[dev]` also installs pytest, which runs the tests. These commands use `pyproject.toml`, matching the main Docker setup.

This guide calls the environment's Python directly, so activating it or changing PowerShell's execution policy is unnecessary. Once setup is complete, you can list the commands:

```powershell
.\.venv\Scripts\python.exe -m src.main --help
```

## 2. Check the input data and output folders

The workflow expects local files. The command-line workflow does not include a data-download step.

Required input locations include:

- `data/raw/f1_circuits_2018_2026_extended.xlsx`
- `data/raw/f1_gp_qualifying_dates_2018_2026.csv`
- Season folders under `data/raw/`, currently covering 2018 through 2026, with GP folders containing session and weather CSV files.

For example, the session-path helper expects a structure like:

```text
data/raw/2026/Australian GP/
  Practice/
    2026-Australian Grand Prix-Practice 1.csv
    2026-Australian Grand Prix-Practice 1-weather.csv
  Qualifying/
    2026-Australian Grand Prix-Qualifying.csv
    2026-Australian Grand Prix-Qualifying-weather.csv
```

Other practice sessions and sprint sessions use the naming rules in `build_session_paths()` in `src/py_def_class.py`. Preserve the existing file names and column structure when adding data.

Some scripts expect output directories to exist already. On a fresh copy, create them with:

```powershell
$outputFolders = @(
    'data/processed', 'data/model_data/train', 'data/model_data/test',
    'data/model_data/final', 'data/data_drift', 'data/predictions',
    'model/params', 'model/eval', 'model/final_model', 'model/offset',
    'model/run_time', 'model/performance', 'model/shap',
    'model/check/train', 'model/check/prediction', 'f1_qualifying_dbt/seeds'
)
foreach ($folder in $outputFolders) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
}
```

Runs can overwrite generated CSV files and saved models. Keep a copy of any results you want to compare with a later run.

Data source and download instructions: **TODO: add the source, collection method, and date collected.**

## 3. Clean and split the data

Run these commands in order:

```powershell
.\.venv\Scripts\python.exe -m src.main quali_time_cleaning
.\.venv\Scripts\python.exe -m src.main train_test_split
```

Cleaning combines the input data and exports `data/processed/Cleaned_quali_f1.csv`, a cleaned circuit file, and a copy of the qualifying dataset in the dbt seeds folder.

Splitting separates the input features (`X`) from the value being predicted (`y`). The target column is `laptime_sum_sectortimes_quali`. The split uses seasons: training ends two seasons before the latest season in the cleaned data, testing uses the preceding season, and the latest season is reserved for later evaluation and drift data. For a latest season of 2026, training ends in 2024 and testing uses 2025.

## 4. Check the prepared data

Run the data checks before expensive training:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_train_test_split_target_check.py tests/test_time_diff_assurance.py -v
```

These tests depend on the files generated in step 3. Review any failure before continuing.

## 5. Train, fit, evaluate, and predict

The individual commands make it easier to see where a problem occurs. Run them in this order, checking the logs after each command:

```powershell
.\.venv\Scripts\python.exe -m src.main all_model_training
.\.venv\Scripts\python.exe -m src.main fitting_final_model
.\.venv\Scripts\python.exe -m src.main eval_final_train_test_model
.\.venv\Scripts\python.exe -m src.main eval_final_model_w_offset
.\.venv\Scripts\python.exe -m src.main making_predictions
```

| Step | What it does |
| --- | --- |
| `all_model_training` | Tunes several model families and saves parameters and scores. This can take substantial time. See the known export issue below. |
| `fitting_final_model` | Fits the final pipeline using saved LightGBM parameters and the combined training/testing dataset. The current code explicitly selects LightGBM. |
| `eval_final_train_test_model` | Evaluates the fitted final model on the later holdout data. |
| `eval_final_model_w_offset` | Calculates and evaluates corrections for systematic prediction errors, saving driver, team, and global offsets. |
| `making_predictions` | Builds prediction inputs, loads the fitted model and offsets, and exports corrected predictions. |

Prediction pauses at `Please enter Compound:`. Enter the intended tyre compound using the categories in your data, for example `SOFT`, and press Enter. The supplied value applies to all rows in that prediction run.

Once individual steps work, this shortcut runs cleaning, splitting, training, fitting, both evaluations, and prediction:

```powershell
.\.venv\Scripts\python.exe -m src.main full_run
```

To run the narrower LightGBM retraining workflow, followed by fitting, evaluation, and prediction:

```powershell
.\.venv\Scripts\python.exe -m src.main final_model_run
```

`final_model_run` also repeats cleaning and splitting. It still performs tuning and may take time. To predict again using existing model, offset, and prepared-data files, run only `making_predictions`.

## 6. Find and assess the results

| Output | Location |
| --- | --- |
| Final fitted model | `model/final_model/final_model_qualifying.joblib` |
| Model parameters | `model/params/` |
| Evaluation scores | `model/eval/` |
| Offset files | `model/offset/` |
| Runtime records | `model/run_time/` |
| Predictions | `data/predictions/2026_qualifying_predictions.csv` |
| Prediction copy for dbt | `f1_qualifying_dbt/seeds/2026_qualifying_predictions.csv` |

The prediction CSV contains `Driver`, `Team`, `GP`, and `pred_corrected`, sorted by GP and corrected predicted time.

When recording results, include the data range, model settings, evaluation period, and comparison with the dummy baseline. MAE measures average absolute error; RMSE gives larger errors more weight; MAPE expresses relative error. Lower values indicate smaller errors on the same evaluation data. Keep any data used to choose corrections separate from data used to claim final performance.

Results summary: **TODO: add your measured scores and what you learned.**

## 7. Run additional tests and analysis

The training smoke test performs model tuning and writes artifacts; it is more expensive than a simple import check. Run it before its output check:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_training_smoke.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_smoke_output_check.py -v
```

Once the required data and smoke-test artifacts exist, the whole suite can be run with:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

For additional analysis after fitting and evaluation:

```powershell
.\.venv\Scripts\python.exe -m src.main data_drift
.\.venv\Scripts\python.exe -m src.main feature_impact_on_pred_shap
```

Drift analysis compares older and newer data. SHAP analysis explains feature contributions and exports `model/shap/feature_impact_on_pred_shap.csv`. These commands are separate from `full_run` and depend on previously generated files.

The notebooks in `notebook/` provide exploratory analysis and development history. A Jupyter environment and notebook-specific packages may be needed; these are separate from the command-line setup above.

## 8. Run with Docker (optional)

Use Docker Desktop with Linux containers and Docker Compose available. Run these commands from the project root after checking the data and creating the output folders:

```powershell
docker compose build app
docker compose run --rm app python -m src.main --help
docker compose run --rm app python -m src.main full_run
```

Keep the terminal interactive so you can answer the compound prompt. The same source-code limitations apply inside Docker.

The `app` service mounts `data/` and `model/`, so changes in those folders persist on your computer. It does **not** mount the dbt seeds folder: seed copies written inside a temporary app container will be removed when that container exits. Before running dbt, copy the persistent outputs into the host seeds folder:

```powershell
Copy-Item -LiteralPath '.\data\processed\Cleaned_quali_f1.csv' -Destination '.\f1_qualifying_dbt\seeds\Cleaned_quali_f1.csv'
Copy-Item -LiteralPath '.\data\predictions\2026_qualifying_predictions.csv' -Destination '.\f1_qualifying_dbt\seeds\2026_qualifying_predictions.csv'
```

Rebuild the app image after changing source code or dependencies.

## 9. Build the dbt data products (optional)

dbt loads the seed CSVs into DuckDB, then uses SQL models to create staging views and data-product tables. Run this after generating or updating the seeds:

```powershell
New-Item -ItemType Directory -Path '.\f1_qualifying_dbt\data' -Force | Out-Null
docker compose build dbt
docker compose run --rm dbt debug
docker compose run --rm dbt seed
docker compose run --rm dbt run
docker compose run --rm dbt test
```

`debug` checks configuration, `seed` loads the CSVs, `run` builds the SQL models, and `test` checks the configured data rules. Stop and inspect failures before proceeding.

The profile uses `/usr/app/data/f1_qualifying.duckdb` inside the container. With the current Compose mount, this persists at `f1_qualifying_dbt/data/f1_qualifying.duckdb` on your computer. Running dbt directly on Windows requires a profile with an appropriate local database path; the supplied profile is configured for Docker.

## Known limitations and troubleshooting

| Symptom or limitation | What to check |
| --- | --- |
| Python 3.11 cannot be found | Install Python 3.11 and confirm `py -3.11 --version` works. |
| `ModuleNotFoundError` | Use `.\.venv\Scripts\python.exe` and install `-e ".[dev]"` from the project root. |
| Relative-import error | Use `-m src.main` as shown above instead of running a source file directly. |
| Missing CSV, model, or parameter file | Check input paths and run the earlier workflow steps that create the file. |
| Cannot save into a missing directory | Create the output folders from step 2. |
| Full training fails while exporting its performance table | `src/all_model_training.py` currently calls `pd.to_csv(df_performance, ...)`. This appears to require `df_performance.to_csv(...)`. Correct and verify that code before relying on `full_run`. |
| Prediction waits or raises an end-of-input error | It requires an interactive answer to the tyre-compound prompt. |
| Smoke-output test fails on a fresh copy | Run the training smoke test first; the output check reads its saved files. |
| Process finishes but expected results are missing | Read the logs for `ERROR` messages and check output timestamps. Some code catches exceptions without re-raising them. |
| New season is not picked up correctly | Review the hardcoded calendar, cleaning range, prediction helper, and filenames. The current code and exports contain explicit 2018-2026/2026 assumptions. |
| dbt uses stale seed data after a Docker app run | Copy the updated CSVs from `data/` into `f1_qualifying_dbt/seeds/`, then repeat the dbt steps. |

## Project maintenance and details to complete

- Record the data source and collection procedure in step 2.
- Add measured results and their evaluation period in step 6.
- Keep dependency changes in `pyproject.toml` and rebuild Docker images when necessary.
- Run the relevant data and model checks after changing preprocessing or training logic.
- When sharing the project, explain how someone can obtain the required datasets.
- Record future improvements below and update this README when commands or paths change.

### License and acknowledgments

The data was gathered from following website:
https://tracinginsights.com/
To be more specific:
https://tracinginsights.com/analysis/download-raw-data/
Here you can select the CSV files for the lap times for free practice 1, free practice 2, free practice 3, Sprint Qualifying, Qualifying and Race.
