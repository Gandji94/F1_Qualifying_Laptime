def eval_final_model_w_offset():
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
    from .py_def_class import Add_Column,RFPermutationRegressorSelector,GroupTimeSplit,previous_stage_param_range,group_time_learning_curve,CastColumnsToObject,abandoned_lap,inspect_quali_time_distribution,f1_rule_era,get_sprint_session_name,build_session_paths,pick_quali_boundaries,practice_quali_new_pred,def_apply_offset_mean,tune_offset_shrinkage_expanding

    #setting up paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA = PROJECT_ROOT / 'data'
    DATA_PROCESSED = DATA/'processed'
    DATA_TRAIN_TEST = DATA/'model_data'
    DATA_FINAL = DATA_TRAIN_TEST/'final'
    DATA_TEST = DATA_TRAIN_TEST/'test'
    DATA_TRAIN = DATA_TRAIN_TEST/'train'
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

    #setting up logging logic
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    #First we will create the offset and evalaute which weighting is the best value
    #Then we will apply the results to the predictions
    logging.info('Loading the avaiable data')
    try:
        df = pd.read_csv(DATA_PROCESSED/'Cleaned_quali_f1.csv')
        #dropping the track temperature columns
        filter_out = set(fo for fo in df.columns if 'TrackTemp' in fo)
        filter_out.add('Qualifying_Date')
        df = df.drop(filter_out,axis=1)
        df = df[df['laptime_sum_sectortimes_quali'].notna()].copy()
    except Exception as ex:
        logging.error(f'An error occurred while loading the data: {ex}')

    logging.info('Loading the final model')
    try:
        final_model = joblib.load(MODEL)
    except Exception as ex:
        logging.error(f'An error occurred while loading the final model')

    logging.info('Creating the offsets')
    try:
        current_season = df['Season'].max()
        print(f"Season used for calculating the Offset: {current_season}")
        df_2026 = df[df['Season'].eq(current_season)]

        gps = np.sort(df_2026["gp_id"].unique())
        available_gps = len(gps)

        if available_gps <= 5:
            logging.info(
                f"Only {available_gps} GPs available. "
                "At least 6 are required."
            )
            return

        frac = 0.60

        offset_length = max(
            5,
            int(np.ceil(available_gps * frac))
        )

        # Keep at least one GP for evaluation
        offset_length = min(
            offset_length,
            available_gps - 1
        )

        offset_gp_id = gps[:offset_length]
        test_offset = gps[offset_length:]


        offset_train_df = df_2026[df_2026['gp_id'].isin(offset_gp_id)]
        off_train_X_train = offset_train_df.drop(['Session','LapTimeDiff_quali','laptime_sum_sectortimes_quali'],axis=1)
        off_train_y_train = offset_train_df['laptime_sum_sectortimes_quali']

        offset_test_df = df_2026[df_2026['gp_id'].isin(test_offset)]
        off_train_X_test = offset_test_df.drop(['Session','LapTimeDiff_quali','laptime_sum_sectortimes_quali'],axis=1)
        off_train_y_test = offset_test_df['laptime_sum_sectortimes_quali']

        pred_offset_df = off_train_X_train.copy()
        off_train_y_pred = final_model.predict(off_train_X_train)
        pred_offset_df['pred'] = off_train_y_pred
        pred_offset_df['actual'] = off_train_y_train
        pred_offset_df['resid'] = pred_offset_df['actual'] - pred_offset_df['pred']

        #driver offset
        driver_offset = pred_offset_df.groupby(['Driver'],as_index=False)['resid'].median().rename(columns={'resid':'driver_offset'})
        driver_offset.to_csv(MODEL_OFFSET/'driver_offset.csv',index=False)
        driver_offset_map = (driver_offset.set_index("Driver")["driver_offset"])

        #team offset
        team_offset = pred_offset_df.groupby(['Team'],as_index=False)['resid'].median().rename(columns={'resid':'team_offset'})
        team_offset.to_csv(MODEL_OFFSET/'team_offset.csv',index=False)
        team_offset_map = (team_offset.set_index("Team")["team_offset"])

        #GP offset
        gp_offset = pred_offset_df.groupby(['GP'],as_index=False)['resid'].median().rename(columns={'resid':'gp_offset'})
        global_gp_offset = gp_offset['gp_offset'].median()
        joblib.dump(global_gp_offset,MODEL_OFFSET/'global_gp_offset.joblib')

        best_shrinking, shrinking_results = (
            tune_offset_shrinkage_expanding(
                calibration_df=pred_offset_df,
                shrinking_values=np.arange(0.0, 1.01, 0.05),
            )
        )

        best_shrink_val = np.round(best_shrinking,2)
        print(f"Best shrinking: {best_shrink_val}")
        joblib.dump(best_shrink_val,MODEL_OFFSET/'best_shrinking_val.joblib')
        print(shrinking_results.head(100))

        df_offset_pred = off_train_X_test.copy()
        y_pred_offset = final_model.predict(off_train_X_test)
        df_offset_pred['Pred_time'] = y_pred_offset

        df_offset_pred["driver_offset"] = (df_offset_pred["Driver"].map(driver_offset_map).fillna(global_gp_offset))
        df_offset_pred["team_offset"] = (df_offset_pred["Team"].map(team_offset_map).fillna(global_gp_offset))
        df_offset_pred['gp_global_offset'] = global_gp_offset
        df_offset_pred['offset'] = df_offset_pred[['driver_offset','team_offset','gp_global_offset']].mean(axis=1)
        df_offset_pred = def_apply_offset_mean(df=df_offset_pred,pred='Pred_time',shrinking=best_shrinking)

        final_mae = mean_absolute_error(off_train_y_test,df_offset_pred['pred_corrected'])
        final_mape = mean_absolute_percentage_error(off_train_y_test,df_offset_pred['pred_corrected'])

        print(f"Final Model MAE: {np.round(final_mae,2)}")
        print(f"Final Model MAPE: {np.round((final_mape*100),2)}%")
    except Exception as ex:
        logging.error(f'An error occurred while evaluating the final model with offset: {ex}')
        raise
    logging.info('Completed')

if __name__ == "__main__":
    eval_final_model_w_offset()