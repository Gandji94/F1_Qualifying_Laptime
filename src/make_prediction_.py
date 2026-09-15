def make_prediction_(
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
        season_max = df['Season'].max()
        df_current_season = df[df['Season'].eq(season_max)]
        print(season_max)
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
        gp_id_lst = []

        # Historical BASE-MODEL residuals.
        # Only GPs with actual qualifying results are added here.
        offset_df = pd.DataFrame(
            columns=[
                "Driver",
                "Team",
                "GP",
                "gp_id",
                "pred",
                "actual",
                "resid",
            ]
        )

        # Store predictions for every GP,
        # including future GPs without actual results.
        final_pred_df = pd.DataFrame()


        for g in np.sort(df_quali["gp_id"].unique()):

            print(f"\nGP: {g}")

            # =========================================================
            # 1. DATA FOR CURRENT GP
            # =========================================================

            prd_df_loop = df_quali[
                df_quali["gp_id"].eq(g)
            ].copy()

            # Always predict the current GP
            quali_pred_values = final_model.predict(
                prd_df_loop
            )

            pred_current = prd_df_loop[
                [
                    "Driver",
                    "Team",
                    "GP",
                    "gp_id",
                ]
            ].copy()

            pred_current["Pred_time"] = quali_pred_values


            # =========================================================
            # 2. CHECK IF ACTUAL QUALIFYING RESULT EXISTS
            # =========================================================

            actual_gp_df = df_current_season[
                df_current_season["gp_id"].eq(g)
            ].copy()

            has_actual = not actual_gp_df.empty

            print(
                f"Historical completed GPs available: "
                f"{len(gp_id_lst)}"
            )

            print(
                f"Actual result available: {has_actual}"
            )


            # =========================================================
            # 3. GP1
            #
            # No historical GP exists yet → no offset
            # =========================================================

            if len(gp_id_lst) == 0:

                final_pred_df_loop = pred_current.copy()

                final_pred_df_loop["driver_offset"] = 0.0
                final_pred_df_loop["team_offset"] = 0.0
                final_pred_df_loop["gp_global_offset"] = 0.0
                final_pred_df_loop["offset"] = 0.0
                final_pred_df_loop["shrinking"] = 0.0

                final_pred_df_loop["pred_corrected"] = (
                    final_pred_df_loop["Pred_time"]
                )

                final_pred_df_loop["applied_offset"] = 0.0


            # =========================================================
            # 4. GP2 AND LATER
            #
            # Use ALL completed previous GPs
            # =========================================================

            else:

                offset_df_loop = offset_df[
                    offset_df["gp_id"].isin(gp_id_lst)
                ].copy()


                # -----------------------------------------------------
                # Driver offset
                # -----------------------------------------------------

                driver_offset = (
                    offset_df_loop
                    .groupby(
                        "Driver",
                        as_index=False
                    )["resid"]
                    .median()
                    .rename(
                        columns={
                            "resid": "driver_offset"
                        }
                    )
                )

                driver_offset_map = (
                    driver_offset
                    .set_index("Driver")["driver_offset"]
                )


                # -----------------------------------------------------
                # Team offset
                # -----------------------------------------------------

                team_offset = (
                    offset_df_loop
                    .groupby(
                        "Team",
                        as_index=False
                    )["resid"]
                    .median()
                    .rename(
                        columns={
                            "resid": "team_offset"
                        }
                    )
                )

                team_offset_map = (
                    team_offset
                    .set_index("Team")["team_offset"]
                )


                # -----------------------------------------------------
                # Global GP offset
                #
                # First calculate median residual for each GP,
                # then take median across those GPs.
                # -----------------------------------------------------

                gp_offset = (
                    offset_df_loop
                    .groupby(
                        "gp_id",
                        as_index=False
                    )["resid"]
                    .median()
                    .rename(
                        columns={
                            "resid": "gp_offset"
                        }
                    )
                )

                global_gp_offset = (
                    gp_offset["gp_offset"].median()
                )


                # -----------------------------------------------------
                # Shrinkage
                #
                # Before GP2 we only have GP1.
                # Expanding-window tuning is impossible with only
                # one historical GP.
                # -----------------------------------------------------

                if len(gp_id_lst) == 1:

                    best_shrinking = 1.0

                    print(
                        f"Only GP {gp_id_lst[0]} available for "
                        f"offset before GP {g}. "
                        f"Using shrinking = {best_shrinking:.2f}"
                    )

                else:

                    best_shrinking, shrinking_results = (
                        tune_offset_shrinkage_expanding(
                            calibration_df=offset_df_loop,
                            shrinking_values=np.arange(
                                0.0,
                                1.01,
                                0.05,
                            ),
                        )
                    )

                    print(
                        f"Best shrinking before GP {g}: "
                        f"{best_shrinking:.2f}"
                    )


                # -----------------------------------------------------
                # Apply offsets to CURRENT GP
                # -----------------------------------------------------

                final_pred_df_loop = pred_current.copy()

                # Driver offset
                final_pred_df_loop["driver_offset"] = (
                    final_pred_df_loop["Driver"]
                    .map(driver_offset_map)
                    .fillna(global_gp_offset)
                )

                # Team offset
                final_pred_df_loop["team_offset"] = (
                    final_pred_df_loop["Team"]
                    .map(team_offset_map)
                    .fillna(global_gp_offset)
                )

                # Global GP offset
                final_pred_df_loop[
                    "gp_global_offset"
                ] = global_gp_offset


                # -----------------------------------------------------
                # Combined offset
                # -----------------------------------------------------

                final_pred_df_loop["offset"] = (
                    final_pred_df_loop[
                        [
                            "driver_offset",
                            "team_offset",
                            "gp_global_offset",
                        ]
                    ]
                    .mean(axis=1)
                )

                final_pred_df_loop[
                    "shrinking"
                ] = best_shrinking


                # -----------------------------------------------------
                # Final correction
                #
                # pred_corrected =
                # Pred_time + shrinking * offset
                # -----------------------------------------------------

                final_pred_df_loop = (
                    def_apply_offset_mean(
                        df=final_pred_df_loop,
                        pred="Pred_time",
                        shrinking=best_shrinking,
                    )
                )


            # =========================================================
            # 5. IF ACTUAL RESULT EXISTS
            #
            # Calculate residual and add current GP to history.
            # =========================================================

            if has_actual:

                actual_current = actual_gp_df[
                    [
                        "Driver",
                        "Team",
                        "GP",
                        "gp_id",
                        "laptime_sum_sectortimes_quali",
                    ]
                ].copy()


                # -----------------------------------------------------
                # Match predictions to actual results
                #
                # Important because prediction and actual dataframes
                # may have different numbers of drivers.
                # -----------------------------------------------------

                history_current = actual_current.merge(
                    pred_current,
                    on=[
                        "Driver",
                        "Team",
                        "GP",
                        "gp_id",
                    ],
                    how="inner",
                )


                # -----------------------------------------------------
                # Data required by offset calculation / tuner
                # -----------------------------------------------------

                history_current["pred"] = (
                    history_current["Pred_time"]
                )

                history_current["actual"] = (
                    history_current[
                        "laptime_sum_sectortimes_quali"
                    ]
                )


                # -----------------------------------------------------
                # Base-model residual
                #
                # actual - prediction
                #
                # Positive:
                # model predicted too fast
                #
                # Negative:
                # model predicted too slow
                # -----------------------------------------------------

                history_current["resid"] = (
                    history_current["actual"]
                    - history_current["pred"]
                )


                # -----------------------------------------------------
                # Add actual to final output
                #
                # LEFT merge keeps predictions even if an actual
                # result is unavailable for a particular driver.
                # -----------------------------------------------------

                actual_for_output = history_current[
                    [
                        "Driver",
                        "Team",
                        "GP",
                        "gp_id",
                        "actual",
                    ]
                ].copy()

                final_pred_df_loop = (
                    final_pred_df_loop
                    .merge(
                        actual_for_output,
                        on=[
                            "Driver",
                            "Team",
                            "GP",
                            "gp_id",
                        ],
                        how="left",
                    )
                )


                # -----------------------------------------------------
                # NOW add the current GP to historical offset data.
                #
                # This happens AFTER prediction, therefore there is
                # no leakage from the current GP.
                # -----------------------------------------------------

                offset_df = pd.concat(
                    [
                        offset_df,
                        history_current[
                            [
                                "Driver",
                                "Team",
                                "GP",
                                "gp_id",
                                "pred",
                                "actual",
                                "resid",
                            ]
                        ],
                    ],
                    axis=0,
                    ignore_index=True,
                )


                # Current GP may be used by NEXT GP
                gp_id_lst.append(g)


            # =========================================================
            # 6. FUTURE GP / NO ACTUAL RESULT
            #
            # Prediction and offsets are still used.
            # GP is simply NOT added to offset history.
            # =========================================================

            else:

                final_pred_df_loop["actual"] = np.nan

                print(
                    f"No actual qualifying result for GP {g}. "
                    f"Prediction created using offsets from "
                    f"{len(gp_id_lst)} previous completed GPs."
                )


            # =========================================================
            # 7. ALWAYS STORE CURRENT GP PREDICTION
            # =========================================================

            final_pred_df = pd.concat(
                [
                    final_pred_df,
                    final_pred_df_loop,
                ],
                axis=0,
                ignore_index=True,
            )


        print("\nCompleted GPs used in offset history:")
        print(gp_id_lst)
    except Exception as ex:
        logging.error(f"An error occurred while creating the prediction output")

    logging.info("Saving the prediction output")
    try:
        prediction_output = final_pred_df[['Driver','Team','GP','pred_corrected']].sort_values(["GP", "pred_corrected"],ascending=[True, True],).reset_index(drop=True)
        prediction_output.to_csv(MODEL_PRED/'2026_qualifying_predictions.csv',index=False)
        prediction_output.to_csv(DATA_DBT_OUTPUT/'2026_qualifying_predictions.csv',index=False)
    except Exception as ex:
        logging.error(f"An error occurred while saving the output")