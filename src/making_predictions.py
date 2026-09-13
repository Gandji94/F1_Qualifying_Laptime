def making_predictions(
        df_filter = None
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import cross_val_score,learning_curve,LearningCurveDisplay
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Lasso,Ridge,ElasticNet
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
    from sklearn.preprocessing import StandardScaler,RobustScaler
    from feature_engine.encoding import MeanEncoder,OneHotEncoder,RareLabelEncoder,CountFrequencyEncoder
    from feature_engine.imputation import AddMissingIndicator,ArbitraryNumberImputer,CategoricalImputer
    from feature_engine.selection import DropFeatures
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from optuna.visualization.matplotlib import plot_param_importances,plot_optimization_history,plot_timeline
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    import seaborn as sns
    import time
    import joblib
    import logging
    from pathlib import Path
    from .py_def_class import Add_Column,RFPermutationRegressorSelector,GroupTimeSplit,previous_stage_param_range,group_time_learning_curve,CastColumnsToObject,abandoned_lap,inspect_quali_time_distribution,f1_rule_era,get_sprint_session_name,build_session_paths,pick_quali_boundaries,practice_quali_new_pred,def_apply_offset_mean,tune_offset_shrinkage_expanding,df_for_prediction

    #setting up paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA = PROJECT_ROOT/'data'
    DATA_RAW = DATA/'raw'
    DATA_PROCESSED = DATA/'processed'
    DATA_TRAIN_TEST = DATA/'model_data'
    DATA_FINAL = DATA_TRAIN_TEST/'final'
    DATA_TEST = DATA_TRAIN_TEST/'test'
    DATA_TRAIN = DATA_TRAIN_TEST/'train'
    DATA_DBT_OUTPUT = PROJECT_ROOT/'f1_qualifying_dbt'/'seeds'

    #model_param
    MODEL_PARAMS = PROJECT_ROOT/'model'/'params'
    ##model
    MODEL = PROJECT_ROOT/'model'/'final_model'/'final_model_qualifying.joblib'
    ##model_eval
    MODEL_EVAL = PROJECT_ROOT/'model'/'eval'
    ##model runtime
    MODEL_RUN_TIME = PROJECT_ROOT/'model'/'run_time'
    #model performance table
    MODEL_PERFORMANCE = PROJECT_ROOT/'model'/'performance'
    #model offset
    MODEL_OFFSET = PROJECT_ROOT/'model'/'offset'
    #model prediction
    MODEL_PRED = DATA/'predictions'

    #setting up logging logic
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info('Loading the avaiable data')
    try:
        df = pd.read_csv(DATA_PROCESSED/'Cleaned_quali_f1.csv')
        #dropping the track temperature columns
        filter_out = set(fo for fo in df.columns if 'TrackTemp' in fo)
        df = df.drop(filter_out,axis=1)
        df = df[df['laptime_sum_sectortimes_quali'].notna()].copy()
    except Exception as ex:
        logging.error(f'An error occurred while loading the data: {ex}')

    logging.info('Loading the final model')
    try:
        final_model = joblib.load(MODEL)
    except Exception as ex:
        logging.error(f'An error occurred while loading the final model')

    logging.info('Creating GP and session information')
    try:
        df_quali = df_for_prediction(
            df_circut=DATA_PROCESSED / 'cleaned_f1_circut.csv',
            quali_data=DATA_RAW,
            df_filter = df_filter
        )
    except Exception as ex:
        logging.error(f'An error occurred while generating GP and session information: {ex}')
        raise

    logging.info('Loading training data')
    try:
        X_train = pd.read_csv(DATA_TRAIN/'X_train.csv')
    except Exception as ex:
        logging.error(f'An error occurred while laoding training data:{ex}')
        raise

    logging.info('Aggregating qualifying features')
    try:
        df_quali = df_quali.drop(['TrackTemp_p1','TrackTemp_p2','TrackTemp_p3','TrackTemp_sprint_quali'],axis=1).copy()

        add_feat = [d for d in df.columns if 'sprint' not in d and ('quali' in d and d not in ['LapTimeDiff_quali','laptime_sum_sectortimes_quali'])]
        # Process every GP independently
        for gp in df_quali["GP"].unique():

            gp_mask = df_quali["GP"].eq(gp)

            df_gp = df_quali.loc[gp_mask].copy()

            # IMPORTANT:
            # Null assessment is now performed only within this GP,
            # like your original single-GP prediction
            null_check_df = df_gp.isnull().sum().to_frame()

            null_lst = set(
                null_check_df[
                    null_check_df[0] > 10
                ].index
            )

            total_set_check = set(
                d for d in df_gp.columns
                if d.startswith(
                    (
                        'AirTemp_',
                        'Humidity_',
                        'Pressure_',
                        'Rainfall_'
                    )
                )
            )

            mean_valid_cols = total_set_check - null_lst

            for a in add_feat:

                sub_mean_cols = [
                    smc
                    for smc in mean_valid_cols
                    if smc.split('_')[0] == a.split('_')[0]
                ]

                if len(sub_mean_cols) > 0:

                    input_add_feat = float(
                        np.mean(
                            [
                                df_gp[x].mean()
                                for x in sub_mean_cols
                            ]
                        )
                    )

                    df_quali.loc[
                        gp_mask,
                        a
                    ] = input_add_feat
        df_quali['Compound_quali'] = input("Please enter Compound: ").strip().upper()
        df_quali = df_quali[[x for x in X_train.columns]].copy()
        #dropping the track temperature columns
        #filter_out = set(fo for fo in df_quali.columns if 'TrackTemp' in fo)
        #df_quali = df_quali.drop(filter_out,axis=1)
    except Exception as ex:
        logging.error(f'An error occurred while aggregating qualifying features: {ex}')
        raise

    logging.info('Making offset predictions')
    try:
        final_model = joblib.load(MODEL)
        quali_pred_values = final_model.predict(df_quali)
        df_quali['Pred_time'] = quali_pred_values

        #driver offset
        driver_offset = pd.read_csv(MODEL_OFFSET/'driver_offset.csv').set_index("Driver")["driver_offset"]
        #team offset
        team_offset = pd.read_csv(MODEL_OFFSET/'team_offset.csv').set_index("Team")["team_offset"]
        #GP offset
        global_gp_offset = joblib.load(MODEL_OFFSET/'global_gp_offset.joblib')


        df_quali["driver_offset"] = (df_quali["Driver"].map(driver_offset).fillna(global_gp_offset))
        df_quali["team_offset"] = (df_quali["Team"].map(team_offset).fillna(global_gp_offset))
        df_quali['gp_global_offset'] = global_gp_offset
        df_quali['offset'] = df_quali[['driver_offset','team_offset','gp_global_offset']].mean(axis=1)
        best_shrinking_val = joblib.load(MODEL_OFFSET/'best_shrinking_val.joblib')
        df_quali = def_apply_offset_mean(df=df_quali,pred='Pred_time',shrinking=best_shrinking_val)
        prediction_output = (df_quali[["Driver", "Team", "GP", "pred_corrected"]].sort_values(["GP", "pred_corrected"],ascending=[True, True],).reset_index(drop=True))
        prediction_output.to_csv(MODEL_PRED/'2026_qualifying_predictions.csv',index=False)
        prediction_output.to_csv(DATA_DBT_OUTPUT/'2026_qualifying_predictions.csv',index=False)

    except Exception as ex:
        logging.error(f'An error occurred while making offset predictions: {ex}')
        raise
    logging.info('Completed')
    print(prediction_output)

if __name__ == "__main__":
    making_predictions()