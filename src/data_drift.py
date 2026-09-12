def data_drift(season_ahead=2):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset
    from sklearn.metrics import mean_absolute_error,root_mean_squared_error,mean_absolute_percentage_error
    import joblib
    import logging
    from pathlib import Path
    from .py_def_class import Add_Column,RFPermutationRegressorSelector,check_data_drift,make_drift_table,get_drift_summary,get_drift_share,historical_drift_backtest,historical_drift_backtest_transformed,historical_drift_backtest_target

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA = PROJECT_ROOT/'data'
    DATA_DRIFT = DATA/'data_drift'
    MODEL = PROJECT_ROOT/'model'/'final_model'/'final_model_qualifying.joblib'

    logging.info('Loading data sets and evaluation scores')
    try:
        X_old = pd.read_csv(DATA_DRIFT/'X_features_old.csv')
        X_new = pd.read_csv(DATA_DRIFT/'X_new.csv')

        y_old = pd.read_csv(DATA_DRIFT/'target_feature.csv')
        y_new = pd.read_csv(DATA_DRIFT/'y_new.csv')

        mae_old = joblib.load(DATA_DRIFT/'final_model_mae_test_performance.joblib')
        rsme_old = joblib.load(DATA_DRIFT/'final_model_rsme_test_performance.joblib')
        mape_old = joblib.load(DATA_DRIFT/'final_model_mape_test_performance.joblib')

        #here we set the min season, what should be the starting point for the histroical analysis of the drift behaviour throughout the seasons
        min_season = X_old['Season'].min()+season_ahead
    except Exception as ex:
        logging.error(f'An error occurred while loading data sets and evaluation scores: {ex}')

    logging.info("Evalauating the raw X feature data's history")
    try:
        exclude_cols = (
            [c for c in X_old.columns if "sprint" in c.lower()]
            + ["Season", "gp_id"]
        )

        drift_cols = [
            c for c in X_old.columns
            if c not in exclude_cols
        ]

        historical_drift = historical_drift_backtest(
            X=X_old,
            start_year=min_season,
            max_round=11,
            exclude_cols=exclude_cols
        )

        print(historical_drift)
    except Exception as ex:
        logging.error(f"An error occured while evalauating the history of the raw X feature data: {ex}")

    logging.info("Getting data drift results for the most recent data")
    try:
        dd_v = historical_drift["drift_share"].describe()
        dd_v['max']

        drift_result = check_data_drift(X_old=X_old[drift_cols],X_new=X_new[drift_cols],drift_share=dd_v['max'].round(2))
        drift_table = make_drift_table(drift_result)
        print(drift_table)
        print('\n')
        print(get_drift_summary(drift_result))
    except Exception as ex:
        logging.error(f"An error occurred while generating the raw X features data drift table: {ex}")

    logging.info("Evalauating the encoded X feature data's history")
    try:
        fm = joblib.load(MODEL)

        step_names = list(fm.named_steps.keys())
        selector_idx = step_names.index("Feature_Selector")

        # Apply everything BEFORE the feature selector
        X_old_encoded = fm[:selector_idx].transform(X_old)
        X_new_encoded = fm[:selector_idx].transform(X_new)

        # Now apply the feature selector
        selector = fm["Feature_Selector"]
        X_old_selected = selector.transform(X_old_encoded)
        X_new_selected = selector.transform(X_new_encoded)

        encoded_cols = set(X_old_selected.columns)
        raw_cols = set(X_old.columns)

        remaining_cols = encoded_cols - raw_cols
        remaining_cols = [c for c in remaining_cols]

        X_old_enc_select = X_old_selected[remaining_cols]
        X_new_enc_select = X_new_selected[remaining_cols]

        #here we run all featured
        historical_encoded_drift = historical_drift_backtest_transformed(
            X_raw=X_old,
            fitted_pipeline=fm,
            start_year=min_season,
            max_round=11
        )
        print(historical_encoded_drift)
        print('\n')
        drift_result = check_data_drift(X_old=X_old_enc_select,X_new=X_new_enc_select,drift_share=0.5)
        drift_table = make_drift_table(drift_result)
        print(drift_table)
        print('\n')
        print(get_drift_summary(drift_result))
    except Exception as ex:
        logging.error(f"An error occurred while generating the encoded X features data drift table: {ex}")

    logging.info("Evaluating the target feature's history")
    try:
        #due to the fact that we only analyse one feature, we do not need to focus on the share
        ##we will use the default on drift_share = 0.5
        target_drift = historical_drift_backtest_target(
            X_data_for_year=X_old,
            y=y_old,
            start_year=min_season,
            max_round=11,
            exclude_cols=exclude_cols
        )
        print(target_drift)
        print('\n')
        drift_result = check_data_drift(X_old=y_old,X_new=y_new,drift_share=0.5)
        drift_table = make_drift_table(drift_result)
        print(drift_table)
    except Exception as ex:
        logging.error(f"An error occurred while evaluating the target feature: {ex}")
    logging.info("Analysing the change in evaluation metrics")
    try:
        new_pred = fm.predict(X_new)
        mae_new = mean_absolute_error(y_new,new_pred)
        rmse_new = root_mean_squared_error(y_new,new_pred)
        mape_new = mean_absolute_percentage_error(y_new,new_pred)

        diff_mae = mae_new - mae_old
        print(f'\n{"Model got significantly worse" if diff_mae > 0.150 else "Current model leads to reliable results"}\n')

        print(f'Old MAE: {mae_old} - New MAE: {mae_new}')
        print(f'Old RSME: {rsme_old} - New RSME: {rmse_new}')
        print(f'Old MAPE: {mape_old} - New RSME: {mape_new}')
    except Exception as ex:
        logging.error(f"An error occurred while analysing/comparing old and new metrics: {ex}")
    logging.info('Completed')