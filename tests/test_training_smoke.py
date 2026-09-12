def test_training_smoke():
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
    import pathlib
    from src.py_def_class import GroupTimeSplit,Add_Column,RFPermutationRegressorSelector,previous_stage_param_range,group_time_learning_curve,CastColumnsToObject

    logging.basicConfig(
        level = logging.INFO,
        format="%(asctime)s [%(levelname)s %(message)s]"
    )

    PROJECT = pathlib.Path(__file__).resolve().parent.parent
    DATA = PROJECT/'data'
    MODEL_DATA = DATA/'model_data'
    MODEL_TRAIN = MODEL_DATA/'train'
    MODEL_TEST = MODEL_DATA/'test'
    MODEL_FINAL = MODEL_DATA/'final'
    MODEL_CHECK = PROJECT/'model'/'check'
    MODEL_CHECK_TRAIN = MODEL_CHECK/'train'
    MODEL_CHECK_PRED = MODEL_CHECK/'prediction'

    logging.info('Loading the training data')
    try:
        #we will import the all X and y features that we have available
        y_train = pd.read_csv(MODEL_TRAIN/'y_train.csv')
        y_test = pd.read_csv(MODEL_TEST/'y_test.csv')
        y_final = pd.read_csv(MODEL_FINAL/'final_y.csv')

        X_train = pd.read_csv(MODEL_TRAIN/'X_train.csv')
        X_test = pd.read_csv(MODEL_TEST/'X_test.csv')
        X_final = pd.read_csv(MODEL_FINAL/'final_X.csv')
    except Exception as ex:
        logging.error(f"An error occurred while loading the training data: {ex}")
        raise

    logging.info('Setting up the smoke test data; Min Season; last three races as test; the rest is training data for the smoke test')
    try:
        min_smoke_season = X_train['Season'].min()
        smoke_train = X_train[X_train['Season'].eq(min_smoke_season)].copy()

        gp_id_train = np.sort(smoke_train['gp_id'].unique())[:-3]
        gp_id_test = np.sort(smoke_train['gp_id'].unique())[-3:]

        smoke_X_train = smoke_train[smoke_train['gp_id'].isin(gp_id_train)].copy()
        smoke_X_train_idx = smoke_X_train.index
        smoke_X_test = smoke_train[smoke_train['gp_id'].isin(gp_id_test)].copy()
        smoke_X_test_idx = smoke_X_test.index

        smoke_y_train = y_train.loc[smoke_X_train_idx].iloc[:,0].copy()
        smoke_y_test = y_train.loc[smoke_X_test_idx].iloc[:,0].copy()
    except Exception as ex:
        logging.error(f"An error occurred while setting up the smoke test data: {ex}")
        raise

    logging.info("Fitting and evaluating the dummy baseline model")
    try:
        for s in ['mean','median']:
            DR = DummyRegressor(strategy=s)
            DR.fit(X_train,y_train)
            y_pred = DR.predict(X_test)

            MAE_DR = mean_absolute_error(y_test,y_pred)
            print(f"DummyRegressor {s} - MAE: {MAE_DR}")
        DR = DummyRegressor(strategy='median')
        DR.fit(smoke_X_train,smoke_y_train)
        y_pred = DR.predict(smoke_X_test)
        MAE_DR = mean_absolute_error(smoke_y_test,y_pred)
        joblib.dump(MAE_DR,MODEL_CHECK_PRED/'prediction_smoke_test_dummy.joblib')
    except Exception as ex:
        logging.error(f"An error occurred while fitting and evaluating the dummy baseline model: {ex}")
        raise

    logging.info("Training, fitting and evaluating the HistGradientBoosting model")
    try:
        def stage1(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []
            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',0.001,0.01,log=True)
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',0.001,0.03,log=True)
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',0.001,0.03,log=True)
            mean_smoothing = trial.suggest_float('mean_smoothing',0.01,10.0,log=True)
            feature_selector = trial.suggest_float('feature_selector',0.0001,0.1,log=True)

            max_iter = trial.suggest_int('max_iter',100,500)
            loss = trial.suggest_categorical('loss',['absolute_error','squared_error'])
            learning_rate = trial.suggest_float('learning_rate',0.02,0.1,log=True)
            max_depth = trial.suggest_int('max_depth',2,4)
            #min_samples_split = trial.suggest_int('min_samples_split',8,32)
            min_samples_leaf = trial.suggest_int('min_samples_leaf',15,50)
            max_leaf_nodes = None
            max_features = trial.suggest_float('max_features',0.6,0.9)
            max_bins = trial.suggest_int('max_bins',50,150)
            l2_regularization = trial.suggest_float('l2_regularization',0.01,10.0,log=True)
            #n_jobs=1

            hgb_param = {
                'max_iter':max_iter,
                'loss':loss,
                'learning_rate':learning_rate,
                'max_depth':max_depth,
                #'min_samples_split':min_samples_split,
                'min_samples_leaf':min_samples_leaf,
                'max_leaf_nodes':max_leaf_nodes,
                'max_features':max_features,
                'max_bins':max_bins,
                'l2_regularization':l2_regularization,
                "early_stopping": False,
                'random_state':101,
                #'n_jobs':n_jobs
            }

            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                #
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(
                        col=mean_freq,
                        name_= ['_mean','_frequency']
                    )),
                    ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RandomForest",HistGradientBoostingRegressor(**hgb_param))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            #print("\nCV MAEs:", mae_fold_lst)
            #print("Mean CV MAE:", np.mean(mae_fold_lst))
            #print("Std CV MAE:", np.std(mae_fold_lst))
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study1=optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=3)
        )

        start_time = time.time()
        study1.optimize(stage1,n_trials=5,n_jobs=-1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study1)
        plt.title("Parameter Importance for Study 1")
        plt.show()

        plot_optimization_history(study1)
        plt.title("Optimazation History for Study 1")
        plt.show()

        plot_timeline(study1)
        plt.title("Timeline for Study 1")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_1_params = study1.best_params
        study_1_best_trial = study1.best_value
        study_1_run_time = end_time - start_time
        print(f"Run time of study 1: {study_1_run_time}")

        #### starting stage 2
        def stage2(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[1])
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[1])
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[1])
            mean_smoothing = trial.suggest_float('mean_smoothing',previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[0],previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[1])
            feature_selector = trial.suggest_float('feature_selector',previous_stage_param_range(study_1_params['feature_selector'],0.3)[0],previous_stage_param_range(study_1_params['feature_selector'],0.3)[1])

            max_iter = trial.suggest_int('max_iter',previous_stage_param_range(study_1_params['max_iter'],0.3)[0],previous_stage_param_range(study_1_params['max_iter'],0.3)[1])
            loss = study_1_params['loss']
            learning_rate = trial.suggest_float('learning_rate',previous_stage_param_range(study_1_params['learning_rate'],0.3)[0],previous_stage_param_range(study_1_params['learning_rate'],0.3)[1])
            max_depth = trial.suggest_int('max_depth',previous_stage_param_range(study_1_params['max_depth'],0.3)[0],previous_stage_param_range(study_1_params['max_depth'],0.3)[1])
            #min_samples_split = trial.suggest_int('min_samples_split',8,32)
            min_samples_leaf = trial.suggest_int('min_samples_leaf',previous_stage_param_range(study_1_params['min_samples_leaf'],0.3)[0],previous_stage_param_range(study_1_params['min_samples_leaf'],0.3)[1])
            max_leaf_nodes = None
            max_features = trial.suggest_float('max_features',previous_stage_param_range(study_1_params['max_features'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['max_features'],0.3,[0.01,1.0])[1])
            max_bins = trial.suggest_int('max_bins',previous_stage_param_range(study_1_params['max_bins'],0.3)[0],previous_stage_param_range(study_1_params['max_bins'],0.3)[1])
            l2_regularization = trial.suggest_float('l2_regularization',previous_stage_param_range(study_1_params['l2_regularization'],0.3)[0],previous_stage_param_range(study_1_params['l2_regularization'],0.3)[1])
            #n_jobs=1

            hgb_param = {
                'max_iter':max_iter,
                'loss':loss,
                'learning_rate':learning_rate,
                'max_depth':max_depth,
                #'min_samples_split':min_samples_split,
                'min_samples_leaf':min_samples_leaf,
                'max_leaf_nodes':max_leaf_nodes,
                'max_features':max_features,
                'max_bins':max_bins,
                'l2_regularization':l2_regularization,
                "early_stopping": False,
                'random_state':101,
                #'n_jobs':n_jobs
            }
            if hgb_param['max_features'] > 1.0:
                hgb_param['max_features'] == 1.0
            elif hgb_param['max_features'] <0.0:
                hgb_param['max_features'] == 0.01

            #adding the fold_idx for the pruner, so the pruner knows, which trial we are
            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                #
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(
                        col=mean_freq,
                        name_= ['_mean','_frequency']
                    )),
                    ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RandomForest",HistGradientBoostingRegressor(**hgb_param))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            #print("\nCV MAEs:", mae_fold_lst)
            #print("Mean CV MAE:", np.mean(mae_fold_lst))
            #print("Std CV MAE:", np.std(mae_fold_lst))
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study2 = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=2)
        )

        start_time = time.time()
        study2.optimize(stage2,n_trials=5,n_jobs=1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study2)
        plt.title("Parameter Importance for Study 2")
        plt.show()

        plot_optimization_history(study2)
        plt.title("Optimization History for Study 2")
        plt.show()

        plot_timeline(study2)
        plt.title("Timeline for Study 2")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_2_params = study2.best_params
        study_2_params['loss'] = study_1_params['loss']
        study_2_params['early_stopping'] = False
        study_2_best_trial = study2.best_value
        study_2_run_time = end_time - start_time
        print(f"Run time of study 2: {study_2_run_time}")

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_param = study_1_params
        else:
            print("Study 2 led to the best result")
            best_param = study_2_params

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_score = study_1_best_trial
        else:
            print("Study 2 led to the best result")
            best_score = study_2_best_trial

        total_time = study_1_run_time + study_2_run_time

        joblib.dump(best_param,MODEL_CHECK_TRAIN/'histgradientboosting_best_param.joblib')
        joblib.dump(best_score,MODEL_CHECK_TRAIN/'histgradientboosting_best_score.joblib')
        joblib.dump(total_time,MODEL_CHECK_TRAIN/'histgradientboosting_run_time.joblib')

        #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
        missing_in_sprint_we = [sprint_not for sprint_not in X_train.columns if 'p2' in sprint_not or 'p3' in sprint_not]
        #
        #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
        mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
        mean_enc_cols = [m+'_mean' for m in mean_freq]
        freq_enc_cols = [f+'_frequency' for f in mean_freq]

        cv=GroupTimeSplit(
            group_column='gp_id',
            n_splits=3,
            test_group_size=3
        )

        hgb_param = {
            'max_iter':best_param['max_iter'],
            'loss':best_param['loss'],
            'learning_rate':best_param['learning_rate'],
            'max_depth':best_param['max_depth'],
            #'min_samples_split':min_samples_split,
            'min_samples_leaf':best_param['min_samples_leaf'],
            'max_leaf_nodes':None,
            'max_features': best_param['max_features'],
            'max_bins':best_param['max_bins'],
            'l2_regularization':best_param['l2_regularization'],
            'early_stopping':False,
            'random_state':101,
            #'n_jobs':n_jobs
        }

        final_hist_pipeline = Pipeline([
            ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=best_param['tol_rare_label_driver'],replace_with='Rare_Driver')),
            ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=best_param['tol_rare_label_gp'],replace_with='Rare_GP')),
            #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
            #because tree-based models can deal with NaN values
            ("RareLabelEncoder_Compounds",RareLabelEncoder(
                variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                tol=best_param['tol_rare_label_compounds'],
                n_categories=2,
                replace_with='Rare_Compound',
                missing_values='ignore'
            )),
            ('Adding_Columns_for_mean_&_frequency',Add_Column(
                col=mean_freq,
                name_= ['_mean','_frequency']
            )),
            ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
            ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
            ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
            ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=best_param['mean_smoothing'],unseen="encode")),
            ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
            ("Feature_Selector",RFPermutationRegressorSelector(threshold=best_param['feature_selector'])),
            ("RandomForest",HistGradientBoostingRegressor(**hgb_param))
        ]).fit(smoke_X_train,smoke_y_train)

        final_hist_pipeline.fit(smoke_X_train,smoke_y_train)
        y_pred = final_hist_pipeline.predict(smoke_X_test)
        hgb_mae = mean_absolute_error(smoke_y_test,y_pred)
        joblib.dump(hgb_mae,MODEL_CHECK_PRED/'prediction_smoke_test_HistGradientBoosting.joblib')
    except Exception as ex:
        logging.error(f"An error occurred while training, fitting and evaluating the HistGradientBoosting model: {ex}")
        raise

    logging.info("Training, fitting and evaluating the Light GBM model")
    try:
        def stage1(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',0.001,0.01,log=True)
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',0.001,0.03,log=True)
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',0.001,0.03,log=True)
            mean_smoothing = trial.suggest_float('mean_smoothing',0.01,10.0,log=True)
            feature_selector = trial.suggest_float('feature_selector',0.0001,0.1,log=True)

            objective = trial.suggest_categorical('objective',['regression','regression_l1','huber'])
            max_bin = trial.suggest_int('max_bin',56,225)
            num_iterations = trial.suggest_int('num_iterations',100,500)
            learning_rate = trial.suggest_float('learning_rate',0.02,0.1,log=True)
            num_leaves = trial.suggest_int('num_leaves',16,64)
            max_depth = trial.suggest_int('max_depth',2,4)
            min_data_in_leaf = trial.suggest_int('min_samples_leaf',5,20)
            min_sum_hessian_in_leaf = trial.suggest_float('min_sum_hessian_in_leaf',0.001,0.1,log=True)
            bagging_fraction = trial.suggest_float('bagging_fraction',0.1,0.9)
            bagging_freq = trial.suggest_int('bagging_freq',1,5)
            colsample_bytree = trial.suggest_float('colsample_bytree',0.01,0.9)
            feature_fraction_bynode = trial.suggest_float('feature_fraction_bynode',0.01,0.9)
            extra_trees = trial.suggest_categorical('extra_trees',[True,False])
            max_delta_step = trial.suggest_float('max_delta_step',0.01,0.9)
            reg_alpha = trial.suggest_float('reg_alpha',0.01,5.0,log=True)
            reg_lambda = trial.suggest_float('reg_lambda',0.01,5.0,log=True)

            lightgbm_params = {
                'objective':objective,
                'max_bin':max_bin,
                'num_iterations':num_iterations,
                'learning_rate':learning_rate,
                'num_leaves':num_leaves,
                'num_threads':1,
                'max_depth':max_depth,
                'min_data_in_leaf':min_data_in_leaf,
                'min_sum_hessian_in_leaf':min_sum_hessian_in_leaf,
                'bagging_fraction': bagging_fraction,
                'bagging_freq':bagging_freq,
                'colsample_bytree':colsample_bytree,
                'feature_fraction_bynode':feature_fraction_bynode,
                'extra_trees':extra_trees,
                'max_delta_step':max_delta_step,
                'reg_alpha':reg_alpha,
                'reg_lambda':reg_lambda,
                'bagging_seed':101,
                'feature_fraction_seed':101,
                'random_state':101,
                'extra_seed':101,
            }

            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                #
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(
                        col=mean_freq,
                        name_= ['_mean','_frequency']
                    )),
                    ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("LightGBM_Regressor",LGBMRegressor(**lightgbm_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study1=optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=3)
        )

        start_time = time.time()
        study1.optimize(stage1,n_trials=5,n_jobs=-1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study1)
        plt.title("Parameter Importance for Study 1")
        plt.show()

        plot_optimization_history(study1)
        plt.title("Optimazation History for Study 1")
        plt.show()

        plot_timeline(study1)
        plt.title("Timeline for Study 1")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_1_params = study1.best_params
        study_1_best_trial = study1.best_value
        study_1_run_time = end_time - start_time
        print(f"Run time of study 1: {study_1_run_time}")

        #### starting stage 2
        def stage2(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[1])
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[1])
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[1])
            mean_smoothing = trial.suggest_float('mean_smoothing',previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[0],previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[1])
            feature_selector = trial.suggest_float('feature_selector',previous_stage_param_range(study_1_params['feature_selector'],0.3)[0],previous_stage_param_range(study_1_params['feature_selector'],0.3)[1])

            objective = study_1_params['objective']
            max_bin = trial.suggest_int('max_bin',previous_stage_param_range(study_1_params['max_bin'],0.3)[0],previous_stage_param_range(study_1_params['max_bin'],0.3)[1])
            num_iterations = trial.suggest_int('num_iterations',previous_stage_param_range(study_1_params['num_iterations'],0.3)[0],previous_stage_param_range(study_1_params['num_iterations'],0.3)[1])
            learning_rate = trial.suggest_float('learning_rate',previous_stage_param_range(study_1_params['learning_rate'],0.3)[0],previous_stage_param_range(study_1_params['learning_rate'],0.3)[1])
            num_leaves = trial.suggest_int('num_leaves',previous_stage_param_range(study_1_params['num_leaves'],0.3)[0],previous_stage_param_range(study_1_params['num_leaves'],0.3)[1])
            max_depth = trial.suggest_int('max_depth',previous_stage_param_range(study_1_params['max_depth'],0.3)[0],previous_stage_param_range(study_1_params['max_depth'],0.3)[1])
            min_data_in_leaf = trial.suggest_int('min_samples_leaf',previous_stage_param_range(study_1_params['min_samples_leaf'],0.3)[0],previous_stage_param_range(study_1_params['min_samples_leaf'],0.3)[1])
            min_sum_hessian_in_leaf = trial.suggest_float('min_sum_hessian_in_leaf',previous_stage_param_range(study_1_params['min_sum_hessian_in_leaf'],0.3)[0],previous_stage_param_range(study_1_params['min_sum_hessian_in_leaf'],0.3)[1])
            bagging_fraction = trial.suggest_float('bagging_fraction',previous_stage_param_range(study_1_params['bagging_fraction'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['bagging_fraction'],0.3,[0.01,1.0])[1])
            bagging_freq = trial.suggest_int('bagging_freq',previous_stage_param_range(study_1_params['bagging_freq'],0.3,)[0],previous_stage_param_range(study_1_params['bagging_freq'],0.3)[1])
            colsample_bytree = trial.suggest_float('colsample_bytree',previous_stage_param_range(study_1_params['colsample_bytree'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['colsample_bytree'],0.3,[0.01,1.0])[1])
            feature_fraction_bynode = trial.suggest_float('feature_fraction_bynode',previous_stage_param_range(study_1_params['feature_fraction_bynode'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['feature_fraction_bynode'],0.3,[0.01,1.0])[1])
            extra_trees = study_1_params['extra_trees']
            max_delta_step = trial.suggest_float('max_delta_step',previous_stage_param_range(study_1_params['max_delta_step'],0.3)[0],previous_stage_param_range(study_1_params['max_delta_step'],0.3)[1])
            reg_alpha = trial.suggest_float('reg_alpha',previous_stage_param_range(study_1_params['reg_alpha'],0.3)[0],previous_stage_param_range(study_1_params['reg_alpha'],0.3)[1])
            reg_lambda = trial.suggest_float('reg_lambda',previous_stage_param_range(study_1_params['reg_lambda'],0.3)[0],previous_stage_param_range(study_1_params['reg_lambda'],0.3)[1])

            lightgbm_params = {
                'objective':objective,
                'max_bin':max_bin,
                'num_iterations':num_iterations,
                'learning_rate':learning_rate,
                'num_leaves':num_leaves,
                'num_threads':1,
                'max_depth':max_depth,
                'min_data_in_leaf':min_data_in_leaf,
                'min_sum_hessian_in_leaf':min_sum_hessian_in_leaf,
                'bagging_fraction': bagging_fraction,
                'bagging_freq':bagging_freq,
                'colsample_bytree':colsample_bytree,
                'feature_fraction_bynode':feature_fraction_bynode,
                'extra_trees':extra_trees,
                'max_delta_step':max_delta_step,
                'reg_alpha':reg_alpha,
                'reg_lambda':reg_lambda,
                'bagging_seed':101,
                'feature_fraction_seed':101,
                'random_state':101,
                'extra_seed':101,
            }

            #adding the fold_idx for the pruner, so the pruner knows, which trial we are
            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                #
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(
                        col=mean_freq,
                        name_= ['_mean','_frequency']
                    )),
                    ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("LightGBM_Regressor",LGBMRegressor(**lightgbm_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study2 = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=2)
        )

        start_time = time.time()
        study2.optimize(stage2,n_trials=5,n_jobs=1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study2)
        plt.title("Parameter Importance for Study 2")
        plt.show()

        plot_optimization_history(study2)
        plt.title("Optimization History for Study 2")
        plt.show()

        plot_timeline(study2)
        plt.title("Timeline for Study 2")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_2_params = study2.best_params
        study_2_params['objective'] = study_1_params['objective']
        study_2_params['extra_trees'] = study_1_params['extra_trees']
        study_2_best_trial = study2.best_value
        study_2_run_time = end_time - start_time
        print(f"Run time of study 2: {study_2_run_time}")

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_param = study_1_params
        else:
            print("Study 2 led to the best result")
            best_param = study_2_params

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_score = study_1_best_trial
        else:
            print("Study 2 led to the best result")
            best_score = study_2_best_trial

        total_time = study_1_run_time + study_2_run_time

        joblib.dump(best_param,MODEL_CHECK_TRAIN/'lightgbm_best_param.joblib')
        joblib.dump(best_score,MODEL_CHECK_TRAIN/'lightgbm_best_score.joblib')
        joblib.dump(total_time,MODEL_CHECK_TRAIN/'lightgbm_run_time.joblib')

        #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
        missing_in_sprint_we = [sprint_not for sprint_not in X_train.columns if 'p2' in sprint_not or 'p3' in sprint_not]
        #
        #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
        mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
        mean_enc_cols = [m+'_mean' for m in mean_freq]
        freq_enc_cols = [f+'_frequency' for f in mean_freq]

        cv=GroupTimeSplit(
            group_column='gp_id',
            n_splits=3,
            test_group_size=3
        )

        lightgbm_params = {
                'objective':best_param['objective'],
                'max_bin':best_param['max_bin'],
                'num_iterations':best_param['num_iterations'],
                'learning_rate':best_param['learning_rate'],
                'num_leaves':best_param['num_leaves'],
                'num_threads':-1,
                'max_depth':best_param['max_depth'],
                'min_data_in_leaf':best_param['min_samples_leaf'],
                'min_sum_hessian_in_leaf':best_param['min_sum_hessian_in_leaf'],
                'bagging_fraction': best_param['bagging_fraction'],
                'bagging_freq': best_param['bagging_freq'],
                'colsample_bytree':best_param['colsample_bytree'],
                'feature_fraction_bynode':best_param['feature_fraction_bynode'],
                'extra_trees':best_param['extra_trees'],
                'max_delta_step':best_param['max_delta_step'],
                'reg_alpha':best_param['reg_alpha'],
                'reg_lambda':best_param['reg_lambda'],
                'bagging_seed':101,
                'feature_fraction_seed':101,
                'random_state':101,
                'extra_seed':101,
        }

        final_hist_pipeline = Pipeline([
            ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=best_param['tol_rare_label_driver'],replace_with='Rare_Driver')),
            ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=best_param['tol_rare_label_gp'],replace_with='Rare_GP')),
            #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
            #because tree-based models can deal with NaN values
            ("RareLabelEncoder_Compounds",RareLabelEncoder(
                variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                tol=best_param['tol_rare_label_compounds'],
                n_categories=2,
                replace_with='Rare_Compound',
                missing_values='ignore'
            )),
            ('Adding_Columns_for_mean_&_frequency',Add_Column(
                col=mean_freq,
                name_= ['_mean','_frequency']
            )),
            ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
            ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
            ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
            ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=best_param['mean_smoothing'],unseen="encode")),
            ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
            ("Feature_Selector",RFPermutationRegressorSelector(threshold=best_param['feature_selector'])),
            ("LightGBM_Regressor",LGBMRegressor(**lightgbm_params))
        ]).fit(smoke_X_train,smoke_y_train)

        final_hist_pipeline.fit(smoke_X_train,smoke_y_train)
        y_pred = final_hist_pipeline.predict(smoke_X_test)
        lightgbm_mae = mean_absolute_error(smoke_y_test,y_pred)
        joblib.dump(lightgbm_mae,MODEL_CHECK_PRED/'prediction_smoke_test_Light_GBM.joblib')
    except Exception as ex:
        logging.error(f"An error occurred while training, fitting and evaluating the Light GBM model")
        raise

    logging.info("Training, fitting and evaluating the XGBoost model")
    try:
        def stage1(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',0.001,0.01,log=True)
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',0.001,0.03,log=True)
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',0.001,0.03,log=True)
            mean_smoothing = trial.suggest_float('mean_smoothing',0.01,10.0,log=True)
            feature_selector = trial.suggest_float('feature_selector',0.0001,0.1,log=True)

            objective = trial.suggest_categorical('objective',['reg:squarederror','reg:absoluteerror'])
            max_depth = trial.suggest_int('max_depth',2,4)
            n_estimators = trial.suggest_int('n_estimators',100,500)
            eta = trial.suggest_float('eta',0.001,0.2,log=True)
            subsample  = trial.suggest_float('subsample',0.01,1.0)
            min_child_weight = trial.suggest_float('min_child_weight',0.1,5.0,log=True)
            max_delta_step = trial.suggest_int('max_delta_step',1,6)
            colsample_bytree = trial.suggest_float('colsample_bytree',0.01,0.9)
            colsample_bylevel = trial.suggest_float('colsample_bylevel',0.01,0.9)
            colsample_bynode = trial.suggest_float('colsample_bynode',0.01,0.9)
            reg_alpha = trial.suggest_float('reg_alpha',0.01,5.0,log=True)
            reg_lambda = trial.suggest_float('reg_lambda',0.01,5.0,log=True)
            #max_leaves = trial.suggest_int('max_leaves',16,48)
            tree_method = trial.suggest_categorical('tree_method',['hist','approx'])
            max_bin = trial.suggest_int('max_bin',32,96)

            xgboost_params = {
                'objective':objective,
                'eval_metric':'mae',
                'max_depth':max_depth,
                'n_estimators':n_estimators,
                'eta':eta,
                'subsample':subsample,
                'min_child_weight':min_child_weight,
                'max_delta_step':max_delta_step,
                'colsample_bytree':colsample_bytree,
                'colsample_bylevel':colsample_bylevel,
                'colsample_bynode':colsample_bynode,
                'reg_alpha':reg_alpha,
                'reg_lambda':reg_lambda,
                'tree_method':tree_method,
                'max_bin':max_bin,
                'random_state':101,
                'n_jobs':1,
                'verbosity':0
                #'max_leaves':max_leaves
            }

            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                #
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(
                        col=mean_freq,
                        name_= ['_mean','_frequency']
                    )),
                    ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("XGBoost_Reg",XGBRegressor(**xgboost_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study1=optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=3)
        )

        start_time = time.time()
        study1.optimize(stage1,n_trials=5,n_jobs=-1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study1)
        plt.title("Parameter Importance for Study 1")
        plt.show()

        plot_optimization_history(study1)
        plt.title("Optimazation History for Study 1")
        plt.show()

        plot_timeline(study1)
        plt.title("Timeline for Study 1")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_1_params = study1.best_params
        study_1_best_trial = study1.best_value
        study_1_run_time = end_time - start_time
        print(f"Run time of study 1: {study_1_run_time}")

        #### starting stage 2
        def stage2(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[1])
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[1])
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[1])
            mean_smoothing = trial.suggest_float('mean_smoothing',previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[0],previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[1])
            feature_selector = trial.suggest_float('feature_selector',previous_stage_param_range(study_1_params['feature_selector'],0.3)[0],previous_stage_param_range(study_1_params['feature_selector'],0.3)[1])

            objective = study_1_params['objective']
            max_depth = trial.suggest_int('max_depth',previous_stage_param_range(study_1_params['max_depth'],0.3)[0],previous_stage_param_range(study_1_params['max_depth'],0.3)[1])
            n_estimators = trial.suggest_int('n_estimators',previous_stage_param_range(study_1_params['n_estimators'],0.3)[0],previous_stage_param_range(study_1_params['n_estimators'],0.3)[1])
            eta = trial.suggest_float('eta',previous_stage_param_range(study_1_params['eta'],0.3)[0],previous_stage_param_range(study_1_params['eta'],0.3)[1])
            subsample = trial.suggest_float('subsample',previous_stage_param_range(study_1_params['subsample'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['subsample'],0.3,[0.01,1.0])[1])
            min_child_weight = trial.suggest_float('min_child_weight',previous_stage_param_range(study_1_params['min_child_weight'],0.3)[0],previous_stage_param_range(study_1_params['min_child_weight'],0.3)[1])
            max_delta_step = trial.suggest_int('max_delta_step',previous_stage_param_range(study_1_params['max_delta_step'],0.3)[0],previous_stage_param_range(study_1_params['max_delta_step'],0.3)[1])
            colsample_bytree = trial.suggest_float('colsample_bytree',previous_stage_param_range(study_1_params['colsample_bytree'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['colsample_bytree'],0.3,[0.01,1.0])[1])
            colsample_bylevel = trial.suggest_float('colsample_bylevel',previous_stage_param_range(study_1_params['colsample_bylevel'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['colsample_bylevel'],0.3,[0.01,1.0])[1])
            colsample_bynode = trial.suggest_float('colsample_bynode',previous_stage_param_range(study_1_params['colsample_bynode'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['colsample_bynode'],0.3,[0.01,1.0])[1])
            reg_alpha = trial.suggest_float('reg_alpha',previous_stage_param_range(study_1_params['reg_alpha'],0.3)[0],previous_stage_param_range(study_1_params['reg_alpha'],0.3)[1])
            reg_lambda = trial.suggest_float('reg_lambda',previous_stage_param_range(study_1_params['reg_lambda'],0.3)[0],previous_stage_param_range(study_1_params['reg_lambda'],0.3)[1])
            #max_leaves = trial.suggest_int('max_leaves',16,48)
            tree_method = study_1_params['tree_method']
            max_bin = trial.suggest_int('max_bin',previous_stage_param_range(study_1_params['max_bin'],0.3)[0],previous_stage_param_range(study_1_params['max_bin'],0.3)[1])

            xgboost_params = {
                'objective':objective,
                'eval_metric':'mae',
                'max_depth':max_depth,
                'n_estimators':n_estimators,
                'eta':eta,
                'subsample':subsample,
                'min_child_weight':min_child_weight,
                'max_delta_step':max_delta_step,
                'colsample_bytree':colsample_bytree,
                'colsample_bylevel':colsample_bylevel,
                'colsample_bynode':colsample_bynode,
                'reg_alpha':reg_alpha,
                'reg_lambda':reg_lambda,
                'tree_method':tree_method,
                'max_bin':max_bin,
                'random_state':101,
                'n_jobs':-1,
                'verbosity':0
                #'max_leaves':max_leaves
            }

            #adding the fold_idx for the pruner, so the pruner knows, which trial we are
            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                #
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(
                        col=mean_freq,
                        name_= ['_mean','_frequency']
                    )),
                    ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("XGBoost_Reg",XGBRegressor(**xgboost_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study2 = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=2)
        )

        start_time = time.time()
        study2.optimize(stage2,n_trials=5,n_jobs=1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study2)
        plt.title("Parameter Importance for Study 2")
        plt.show()

        plot_optimization_history(study2)
        plt.title("Optimization History for Study 2")
        plt.show()

        plot_timeline(study2)
        plt.title("Timeline for Study 2")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_2_params = study2.best_params
        study_2_params['objective'] = study_1_params['objective']
        study_2_params['tree_method'] = study_1_params['tree_method']
        study_2_best_trial = study2.best_value
        study_2_run_time = end_time - start_time
        print(f"Run time of study 2: {study_2_run_time}")

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_param = study_1_params
            best_score = study_1_best_trial
        else:
            print("Study 2 led to the best result")
            best_param = study_2_params
            best_score = study_2_best_trial

        total_time = study_1_run_time + study_2_run_time

        joblib.dump(best_param,MODEL_CHECK_TRAIN/'xgboost_best_param.joblib')
        joblib.dump(best_score,MODEL_CHECK_TRAIN/'xgboost_best_score.joblib')
        joblib.dump(total_time,MODEL_CHECK_TRAIN/'xgboost_run_time.joblib')

        #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
        missing_in_sprint_we = [sprint_not for sprint_not in X_train.columns if 'p2' in sprint_not or 'p3' in sprint_not]
        #
        #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
        mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
        mean_enc_cols = [m+'_mean' for m in mean_freq]
        freq_enc_cols = [f+'_frequency' for f in mean_freq]

        cv=GroupTimeSplit(
            group_column='gp_id',
            n_splits=3,
            test_group_size=3
        )

        xgboost_params = {
                'objective':best_param['objective'],
                'eval_metric':'mae',
                'max_depth':best_param['max_depth'],
                'n_estimators':best_param['n_estimators'],
                'eta':best_param['eta'],
                'subsample':best_param['subsample'],
                'min_child_weight':best_param['min_child_weight'],
                'max_delta_step':best_param['max_delta_step'],
                'colsample_bytree':best_param['colsample_bytree'],
                'colsample_bylevel':best_param['colsample_bylevel'],
                'colsample_bynode':best_param['colsample_bynode'],
                'reg_alpha':best_param['reg_alpha'],
                'reg_lambda':best_param['reg_lambda'],
                'tree_method':best_param['tree_method'],
                'max_bin':best_param['max_bin'],
                'random_state':101,
                'n_jobs':-1,
                'verbosity':0
                #'max_leaves':max_leaves
            }

        final_hist_pipeline = Pipeline([
            ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=best_param['tol_rare_label_driver'],replace_with='Rare_Driver')),
            ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=best_param['tol_rare_label_gp'],replace_with='Rare_GP')),
            #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
            #because tree-based models can deal with NaN values
            ("RareLabelEncoder_Compounds",RareLabelEncoder(
                variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                tol=best_param['tol_rare_label_compounds'],
                n_categories=2,
                replace_with='Rare_Compound',
                missing_values='ignore'
            )),
            ('Adding_Columns_for_mean_&_frequency',Add_Column(
                col=mean_freq,
                name_= ['_mean','_frequency']
            )),
            ("Indicating_Missing_Value",AddMissingIndicator(missing_only=True,variables=missing_in_sprint_we)),
            ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
            ('One_Hot_Encoded_Features',OneHotEncoder(variables=['Pace_profile','Type','Direction','Rule_Era'])),
            ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=best_param['mean_smoothing'],unseen="encode")),
            ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
            ("Feature_Selector",RFPermutationRegressorSelector(threshold=best_param['feature_selector'])),
            ("XGBoost_Reg",XGBRegressor(**xgboost_params))
        ]).fit(smoke_X_train,smoke_y_train)

        final_hist_pipeline.fit(smoke_X_train,smoke_y_train)
        y_pred = final_hist_pipeline.predict(smoke_X_test)
        xgboost_mae = mean_absolute_error(smoke_y_test,y_pred)
        joblib.dump(xgboost_mae,MODEL_CHECK_PRED/'prediction_smoke_test_XGBoost.joblib')

    except Exception as ex:
        logging.error(f"An error occurred while training, fitting and evaluating the XGBoost model")
        raise

    logging.info("Training, fitting and evaluating the MLP model")
    try:
        def stage1(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',0.001,0.01,log=True)
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',0.001,0.03,log=True)
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',0.001,0.03,log=True)
            mean_smoothing = trial.suggest_float('mean_smoothing',0.01,10.0,log=True)
            feature_selector = trial.suggest_float('feature_selector',0.0001,0.1,log=True)

            hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes',[(8,16),(32,16),(64,32)])
            activation = trial.suggest_categorical('activation',["relu", "tanh"])
            solver = trial.suggest_categorical('solver',["adam", "lbfgs"])
            alpha = trial.suggest_float('alpha',0.0001,0.9)
            learning_rate_init = trial.suggest_float('learning_rate_init',0.0001,0.9)
            early_stopping = True
            validation_fraction = trial.suggest_float('validation_fraction',0.1,0.3)
            epsilon = trial.suggest_float('epsilon',0.00000001,0.01)
            max_iter = trial.suggest_int('max_iter',250,750)
            n_iter_no_change = trial.suggest_int('n_iter_no_change',2,10)
            

            mlp_params = {
                'hidden_layer_sizes':hidden_layer_sizes,
                'activation':activation,
                'solver':solver,
                'alpha':alpha,
                'learning_rate_init':learning_rate_init,
                'early_stopping':early_stopping,
                'validation_fraction':validation_fraction,
                'epsilon':epsilon,
                'max_iter':max_iter,
                'n_iter_no_change':n_iter_no_change,
                'random_state':101
            }

            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                missing_input_features = [ip for ip in X_train_.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
                df_missing = X_train_[missing_input_features]
                missing_num = list(df_missing.select_dtypes(include=['number']).columns)
                missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)
                
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

                cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
                    ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
                    ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
                    ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
                    ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
                    ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
                    ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RobustScaler",RobustScaler()),
                    ("MLP_Reg",MLPRegressor(**mlp_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study1=optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=3)
        )

        start_time = time.time()
        study1.optimize(stage1,n_trials=5,n_jobs=-1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study1)
        plt.title("Parameter Importance for Study 1")
        plt.show()

        plot_optimization_history(study1)
        plt.title("Optimazation History for Study 1")
        plt.show()

        plot_timeline(study1)
        plt.title("Timeline for Study 1")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_1_params = study1.best_params
        study_1_params['early_stopping'] = True

        study_1_best_trial = study1.best_value
        study_1_run_time = end_time - start_time
        print(f"Run time of study 1: {study_1_run_time}")

        #### starting stage 2
        def stage2(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[1])
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[1])
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[1])
            mean_smoothing = trial.suggest_float('mean_smoothing',previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[0],previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[1])
            feature_selector = trial.suggest_float('feature_selector',previous_stage_param_range(study_1_params['feature_selector'],0.3)[0],previous_stage_param_range(study_1_params['feature_selector'],0.3)[1])

            hidden_layer_sizes = study_1_params['hidden_layer_sizes']
            activation = study_1_params['activation']
            solver = study_1_params['solver']
            alpha = trial.suggest_float('alpha',previous_stage_param_range(study_1_params['alpha'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['alpha'],0.3,[0.01,1.0])[1])
            learning_rate_init = trial.suggest_float('learning_rate_init',previous_stage_param_range(study_1_params['learning_rate_init'],0.3)[0],previous_stage_param_range(study_1_params['learning_rate_init'],0.3)[1])
            early_stopping = study_1_params['early_stopping']
            validation_fraction = trial.suggest_float('validation_fraction',previous_stage_param_range(study_1_params['validation_fraction'],0.3,[0.01,1.0])[0],previous_stage_param_range(study_1_params['validation_fraction'],0.3,[0.01,1.0])[1])
            epsilon = trial.suggest_float('epsilon',previous_stage_param_range(study_1_params['epsilon'],0.3)[0],previous_stage_param_range(study_1_params['epsilon'],0.3)[1])
            max_iter = trial.suggest_int('max_iter',previous_stage_param_range(study_1_params['max_iter'],0.3)[0],previous_stage_param_range(study_1_params['max_iter'],0.3)[1])
            n_iter_no_change = trial.suggest_int('n_iter_no_change',previous_stage_param_range(study_1_params['n_iter_no_change'],0.3)[0],previous_stage_param_range(study_1_params['n_iter_no_change'],0.3)[1])
            
            mlp_params = {
                'hidden_layer_sizes':hidden_layer_sizes,
                'activation':activation,
                'solver':solver,
                'alpha':alpha,
                'learning_rate_init':learning_rate_init,
                'early_stopping':early_stopping,
                'validation_fraction':validation_fraction,
                'epsilon':epsilon,
                'max_iter':max_iter,
                'n_iter_no_change':n_iter_no_change,
                'random_state':101
            }

            #adding the fold_idx for the pruner, so the pruner knows, which trial we are
            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                missing_input_features = [ip for ip in X_train_.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
                df_missing = X_train_[missing_input_features]
                missing_num = list(df_missing.select_dtypes(include=['number']).columns)
                missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)
                
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

                cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
                    ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
                    ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
                    ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
                    ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
                    ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
                    ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RobustScaler",RobustScaler()),
                    ("MLP_Reg",MLPRegressor(**mlp_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study2 = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=10,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=2)
        )

        start_time = time.time()
        study2.optimize(stage2,n_trials=5,n_jobs=1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study2)
        plt.title("Parameter Importance for Study 2")
        plt.show()

        plot_optimization_history(study2)
        plt.title("Optimization History for Study 2")
        plt.show()

        plot_timeline(study2)
        plt.title("Timeline for Study 2")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_2_params = study2.best_params
        study_2_params['hidden_layer_sizes'] = study_1_params['hidden_layer_sizes']
        study_2_params['activation'] = study_1_params['activation']
        study_2_params['solver'] = study_1_params['solver']
        study_2_params['early_stopping'] = study_1_params['early_stopping']
        study_2_best_trial = study2.best_value
        study_2_run_time = end_time - start_time
        print(f"Run time of study 2: {study_2_run_time}")

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_param = study_1_params
            best_score = study_1_best_trial
        else:
            print("Study 2 led to the best result")
            best_param = study_2_params
            best_score = study_2_best_trial

        total_time = study_1_run_time + study_2_run_time

        joblib.dump(best_param,MODEL_CHECK_TRAIN/'mlp_best_param.joblib')
        joblib.dump(best_score,MODEL_CHECK_TRAIN/'mlp_best_score.joblib')
        joblib.dump(total_time,MODEL_CHECK_TRAIN/'mlp_run_time.joblib')

        missing_input_features = [ip for ip in X_train.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
        df_missing = X_train[missing_input_features]
        missing_num = list(df_missing.select_dtypes(include=['number']).columns)
        missing_ob = list(df_missing.select_dtypes(include=['object','category']).columns)

        #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
        mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
        mean_enc_cols = [m+'_mean' for m in mean_freq]
        freq_enc_cols = [f+'_frequency' for f in mean_freq]

        one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

        cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

        cv=GroupTimeSplit(
            group_column='gp_id',
            n_splits=3,
            test_group_size=3
        )

        mlp_params = {
            'hidden_layer_sizes':best_param['hidden_layer_sizes'],
            'activation':best_param['activation'],
            'solver':best_param['solver'],
            'alpha':best_param['alpha'],
            'learning_rate_init':best_param['learning_rate_init'],
            'early_stopping':best_param['early_stopping'],
            'validation_fraction':best_param['validation_fraction'],
            'epsilon':best_param['epsilon'],
            'max_iter':best_param['max_iter'],
            'n_iter_no_change':best_param['n_iter_no_change'],
            'random_state':101
        }

        final_hist_pipeline = Pipeline([
            ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=best_param['tol_rare_label_driver'],replace_with='Rare_Driver')),
            ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=best_param['tol_rare_label_gp'],replace_with='Rare_GP')),
            #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
            #because tree-based models can deal with NaN values
            ("RareLabelEncoder_Compounds",RareLabelEncoder(
                variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                tol=best_param['tol_rare_label_compounds'],
                n_categories=2,
                replace_with='Rare_Compound',
                missing_values='ignore'
            )),
            ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
            ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
            ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
            ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
            ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
            ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
            ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
            ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
            ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
            ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=best_param['mean_smoothing'],unseen="encode")),
            ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
            #("Feature_Selector",RFPermutationRegressorSelector(threshold=best_param['feature_selector'])),
            ("Feature_Selector",RFPermutationRegressorSelector(threshold=0.0)),
            ("RobustScaler",RobustScaler()),
            ("MLP_Reg",MLPRegressor(**mlp_params))
        ]).fit(smoke_X_train,smoke_y_train)

        final_hist_pipeline.fit(smoke_X_train,smoke_y_train)
        y_pred = final_hist_pipeline.predict(smoke_X_test)
        mlp_mae = mean_absolute_error(smoke_y_test,y_pred)
        joblib.dump(mlp_mae,MODEL_CHECK_PRED/'prediction_smoke_test_MLP.joblib')
    except Exception as ex:
        logging.error(f"An error occurred while training, fitting and evaluating the MLP model")
        raise

    logging.info("Training, fitting and evaluating the Lasso model")
    try:
        def stage1(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',0.001,0.01,log=True)
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',0.001,0.03,log=True)
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',0.001,0.03,log=True)
            mean_smoothing = trial.suggest_float('mean_smoothing',0.01,10.0,log=True)
            feature_selector = trial.suggest_float('feature_selector',0.0001,0.1,log=True)

            alpha = trial.suggest_float('alpha',0.0001,0.01)
            max_iter = trial.suggest_int('max_iter',1000,5000)
            tol = trial.suggest_float('tol',0.0001,0.01)
            selection = trial.suggest_categorical('selection',['cyclic','random'])

            lasso_params = {
                'alpha':alpha,
                'max_iter':max_iter,
                'tol':tol,
                'selection':selection,
                'random_state':101
            }

            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                #missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                missing_input_features = [ip for ip in X_train_.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
                df_missing = X_train_[missing_input_features]
                missing_num = list(df_missing.select_dtypes(include=['number']).columns)
                missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)
                
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

                cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
                    ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
                    ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
                    ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
                    ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
                    ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
                    ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RobustScaler",RobustScaler()),
                    ("Lasso_Regressor",Lasso(**lasso_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study1=optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=3)
        )

        start_time = time.time()
        study1.optimize(stage1,n_trials=5,n_jobs=-1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study1)
        plt.title("Parameter Importance for Study 1")
        plt.show()

        plot_optimization_history(study1)
        plt.title("Optimazation History for Study 1")
        plt.show()

        plot_timeline(study1)
        plt.title("Timeline for Study 1")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_1_params = study1.best_params
        study_1_best_trial = study1.best_value
        study_1_run_time = end_time - start_time
        print(f"Run time of study 1: {study_1_run_time}")

        #### starting stage 2
        def stage2(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[1])
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[1])
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[1])
            mean_smoothing = trial.suggest_float('mean_smoothing',previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[0],previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[1])
            feature_selector = trial.suggest_float('feature_selector',previous_stage_param_range(study_1_params['feature_selector'],0.3)[0],previous_stage_param_range(study_1_params['feature_selector'],0.3)[1])

            alpha = trial.suggest_float('alpha',previous_stage_param_range(study_1_params['alpha'],0.3,[0.001,1.0])[0],previous_stage_param_range(study_1_params['alpha'],0.3,[0.001,1.0])[1])
            max_iter = trial.suggest_int('max_iter',previous_stage_param_range(study_1_params['max_iter'],0.3)[0],previous_stage_param_range(study_1_params['max_iter'],0.3)[1])
            tol = trial.suggest_float('tol',previous_stage_param_range(study_1_params['tol'],0.3)[0],previous_stage_param_range(study_1_params['tol'],0.3)[1])
            selection = study_1_params['selection']

            lasso_params = {
                'alpha':alpha,
                'max_iter':max_iter,
                'tol':tol,
                'selection':selection,
                'random_state':101
            }

            #adding the fold_idx for the pruner, so the pruner knows, which trial we are
            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                missing_input_features = [ip for ip in X_train_.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
                df_missing = X_train_[missing_input_features]
                missing_num = list(df_missing.select_dtypes(include=['number']).columns)
                missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)
                
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

                cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
                    ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
                    ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
                    ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
                    ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
                    ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
                    ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RobustScaler",RobustScaler()),
                    ("Lasso_Regressor",Lasso(**lasso_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study2 = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=2)
        )

        start_time = time.time()
        study2.optimize(stage2,n_trials=5,n_jobs=1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study2)
        plt.title("Parameter Importance for Study 2")
        plt.show()

        plot_optimization_history(study2)
        plt.title("Optimization History for Study 2")
        plt.show()

        plot_timeline(study2)
        plt.title("Timeline for Study 2")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_2_params = study2.best_params
        study_2_params['selection'] = study_1_params['selection']
        study_2_best_trial = study2.best_value
        study_2_run_time = end_time - start_time
        print(f"Run time of study 2: {study_2_run_time}")

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_param = study_1_params
        else:
            print("Study 2 led to the best result")
            best_param = study_2_params

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_score = study_1_best_trial
        else:
            print("Study 2 led to the best result")
            best_score = study_2_best_trial

        total_time = study_1_run_time + study_2_run_time

        joblib.dump(best_param,MODEL_CHECK_TRAIN/'lasso_best_param.joblib')
        joblib.dump(best_score,MODEL_CHECK_TRAIN/'lasso_best_score.joblib')
        joblib.dump(total_time,MODEL_CHECK_TRAIN/'lasso_run_time.joblib')

        missing_input_features = [ip for ip in X_train.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
        df_missing = X_train[missing_input_features]
        missing_num = list(df_missing.select_dtypes(include=['number']).columns)
        missing_ob = list(df_missing.select_dtypes(include=['object','category']).columns)

        #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
        mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
        mean_enc_cols = [m+'_mean' for m in mean_freq]
        freq_enc_cols = [f+'_frequency' for f in mean_freq]

        one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

        cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

        cv=GroupTimeSplit(
            group_column='gp_id',
            n_splits=3,
            test_group_size=3
        )

        lasso_params = {
                'alpha':best_param['alpha'],
                'max_iter':best_param['max_iter'],
                'tol':best_param['tol'],
                'selection':best_param['selection'],
                'random_state':101
            }

        final_hist_pipeline = Pipeline([
            ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=best_param['tol_rare_label_driver'],replace_with='Rare_Driver')),
            ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=best_param['tol_rare_label_gp'],replace_with='Rare_GP')),
            #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
            #because tree-based models can deal with NaN values
            ("RareLabelEncoder_Compounds",RareLabelEncoder(
                variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                tol=best_param['tol_rare_label_compounds'],
                n_categories=2,
                replace_with='Rare_Compound',
                missing_values='ignore'
            )),
            ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
            ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
            ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
            ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
            ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
            ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
            ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
            ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
            ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
            ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=best_param['mean_smoothing'],unseen="encode")),
            ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
            ("Feature_Selector",RFPermutationRegressorSelector(threshold=best_param['feature_selector'])),
            ("RobustScaler",RobustScaler()),
            ("Lasso_Regressor",Lasso(**lasso_params))
        ]).fit(smoke_X_train,smoke_y_train)

        final_hist_pipeline.fit(smoke_X_train,smoke_y_train)
        y_pred = final_hist_pipeline.predict(smoke_X_test)
        lasso_mae = mean_absolute_error(smoke_y_test,y_pred)
        joblib.dump(lasso_mae,MODEL_CHECK_PRED/'prediction_smoke_test_Lasso.joblib')
    except Exception as ex:
        logging.error(f"An error occurre while training, fitting and evaluating the Lasso model")
        raise

    logging.info("Training, fitting and evaluating the Ridge model")
    try:
        def stage1(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',0.001,0.01,log=True)
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',0.001,0.03,log=True)
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',0.001,0.03,log=True)
            mean_smoothing = trial.suggest_float('mean_smoothing',0.01,10.0,log=True)
            feature_selector = trial.suggest_float('feature_selector',0.0001,0.1,log=True)

            alpha = trial.suggest_float('alpha',0.00001,0.1)
            max_iter = trial.suggest_int('max_iter',1000,5000)
            tol = trial.suggest_float('tol',0.0001,0.01)

            ridge_params = {
                'alpha':alpha,
                'max_iter':max_iter,
                'tol':tol,
                'random_state':101
            }

            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                #missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                missing_input_features = [ip for ip in X_train_.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
                df_missing = X_train_[missing_input_features]
                missing_num = list(df_missing.select_dtypes(include=['number']).columns)
                missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)
                
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

                cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
                    ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
                    ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
                    ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
                    ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
                    ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
                    ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RobustScaler",RobustScaler()),
                    ("Ridge_Regressor",Ridge(**ridge_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study1=optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=3)
        )

        start_time = time.time()
        study1.optimize(stage1,n_trials=5,n_jobs=-1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study1)
        plt.title("Parameter Importance for Study 1")
        plt.show()

        plot_optimization_history(study1)
        plt.title("Optimazation History for Study 1")
        plt.show()

        plot_timeline(study1)
        plt.title("Timeline for Study 1")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_1_params = study1.best_params
        study_1_best_trial = study1.best_value
        study_1_run_time = end_time - start_time
        print(f"Run time of study 1: {study_1_run_time}")

        #### starting stage 2
        def stage2(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[1])
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[1])
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[1])
            mean_smoothing = trial.suggest_float('mean_smoothing',previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[0],previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[1])
            feature_selector = trial.suggest_float('feature_selector',previous_stage_param_range(study_1_params['feature_selector'],0.3)[0],previous_stage_param_range(study_1_params['feature_selector'],0.3)[1])

            alpha = trial.suggest_float('alpha',previous_stage_param_range(study_1_params['alpha'],0.3,[0.001,1.0])[0],previous_stage_param_range(study_1_params['alpha'],0.3,[0.001,1.0])[1])
            max_iter = trial.suggest_int('max_iter',previous_stage_param_range(study_1_params['max_iter'],0.3)[0],previous_stage_param_range(study_1_params['max_iter'],0.3)[1])
            tol = trial.suggest_float('tol',previous_stage_param_range(study_1_params['tol'],0.3)[0],previous_stage_param_range(study_1_params['tol'],0.3)[1])
            #selection = study_1_params['selection']

            ridge_params = {
                'alpha':alpha,
                'max_iter':max_iter,
                'tol':tol,
                'random_state':101
            }

            #adding the fold_idx for the pruner, so the pruner knows, which trial we are
            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                #missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                missing_input_features = [ip for ip in X_train_.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
                df_missing = X_train_[missing_input_features]
                missing_num = list(df_missing.select_dtypes(include=['number']).columns)
                missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)
                
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

                cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
                    ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
                    ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
                    ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
                    ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
                    ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
                    ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RobustScaler",RobustScaler()),
                    ("Ridge_Regressor",Ridge(**ridge_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study2 = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=2)
        )

        start_time = time.time()
        study2.optimize(stage2,n_trials=5,n_jobs=1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study2)
        plt.title("Parameter Importance for Study 2")
        plt.show()

        plot_optimization_history(study2)
        plt.title("Optimization History for Study 2")
        plt.show()

        plot_timeline(study2)
        plt.title("Timeline for Study 2")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_2_params = study2.best_params
        #study_2_params['selection'] = study_1_params['selection']
        study_2_best_trial = study2.best_value
        study_2_run_time = end_time - start_time
        print(f"Run time of study 2: {study_2_run_time}")

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_param = study_1_params
        else:
            print("Study 2 led to the best result")
            best_param = study_2_params

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_score = study_1_best_trial
        else:
            print("Study 2 led to the best result")
            best_score = study_2_best_trial

        total_time = study_1_run_time + study_2_run_time

        joblib.dump(best_param,MODEL_CHECK_TRAIN/'ridge_best_param.joblib')
        joblib.dump(best_score,MODEL_CHECK_TRAIN/'ridge_best_score.joblib')
        joblib.dump(total_time,MODEL_CHECK_TRAIN/'ridge_run_time.joblib')

        missing_input_features = [ip for ip in X_train.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
        df_missing = X_train[missing_input_features]
        missing_num = list(df_missing.select_dtypes(include=['number']).columns)
        missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)

        #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
        mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
        mean_enc_cols = [m+'_mean' for m in mean_freq]
        freq_enc_cols = [f+'_frequency' for f in mean_freq]

        one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

        cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))


        cv=GroupTimeSplit(
            group_column='gp_id',
            n_splits=3,
            test_group_size=3
        )

        ridge_params = {
                'alpha':best_param['alpha'],
                'max_iter':best_param['max_iter'],
                'tol':best_param['tol'],
                'random_state':101
            }

        final_hist_pipeline = Pipeline([
            ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=best_param['tol_rare_label_driver'],replace_with='Rare_Driver')),
            ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=best_param['tol_rare_label_gp'],replace_with='Rare_GP')),
            #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
            #because tree-based models can deal with NaN values
            ("RareLabelEncoder_Compounds",RareLabelEncoder(
                variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                tol=best_param['tol_rare_label_compounds'],
                n_categories=2,
                replace_with='Rare_Compound',
                missing_values='ignore'
            )),
            ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
            ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
            ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
            ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
            ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
            ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
            ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
            ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
            ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
            ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=best_param['mean_smoothing'],unseen="encode")),
            ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
            ("Feature_Selector",RFPermutationRegressorSelector(threshold=best_param['feature_selector'])),
            ("RobustScaler",RobustScaler()),
            ("Ridge_Regressor",Ridge(**ridge_params))
        ]).fit(smoke_X_train,smoke_y_train)

        final_hist_pipeline.fit(smoke_X_train,smoke_y_train)
        y_pred = final_hist_pipeline.predict(smoke_X_test)
        ridge_mae = mean_absolute_error(smoke_y_test,y_pred)
        joblib.dump(ridge_mae,MODEL_CHECK_PRED/'prediction_smoke_test_Ridge.joblib')
    except Exception as ex:
        logging.error(f"An error occurre while training, fitting and evaluating the Ridge model")
        raise

    logging.info("Training, fitting and evaluating the Elastic Net model")
    try:
        def stage1(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',0.001,0.01,log=True)
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',0.001,0.03,log=True)
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',0.001,0.03,log=True)
            mean_smoothing = trial.suggest_float('mean_smoothing',0.01,10.0,log=True)
            feature_selector = trial.suggest_float('feature_selector',0.0001,0.1,log=True)

            alpha = trial.suggest_float('alpha',0.00001,0.1)
            l1_ratio = trial.suggest_float('l1_ratio',0.001,0.5)
            max_iter = trial.suggest_int('max_iter',1000,5000)
            tol = trial.suggest_float('tol',0.00001,0.001)
            selection = trial.suggest_categorical('selection',['cyclic','random'])

            elastic_net_params = {
                'alpha':alpha,
                'l1_ratio':l1_ratio,
                'max_iter':max_iter,
                'tol':tol,
                'selection':selection,
                'random_state':101
            }

            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                #missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                missing_input_features = [ip for ip in X_train_.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
                df_missing = X_train_[missing_input_features]
                missing_num = list(df_missing.select_dtypes(include=['number']).columns)
                missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)
                
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

                cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
                    ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
                    ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
                    ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
                    ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
                    ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
                    ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RobustScaler",RobustScaler()),
                    ("Elastic_Net_Regressor",ElasticNet(**elastic_net_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study1=optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=3)
        )

        start_time = time.time()
        study1.optimize(stage1,n_trials=5,n_jobs=-1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study1)
        plt.title("Parameter Importance for Study 1")
        plt.show()

        plot_optimization_history(study1)
        plt.title("Optimazation History for Study 1")
        plt.show()

        plot_timeline(study1)
        plt.title("Timeline for Study 1")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_1_params = study1.best_params
        study_1_best_trial = study1.best_value
        study_1_run_time = end_time - start_time
        print(f"Run time of study 1: {study_1_run_time}")

        #### starting stage 2
        def stage2(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=3,
                test_group_size=3
            )
            
            mae_fold_lst = []

            tol_rare_label_driver = trial.suggest_float('tol_rare_label_driver',previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_driver'],0.3)[1])
            tol_rare_label_gp = trial.suggest_float('tol_rare_label_gp',previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_gp'],0.3)[1])
            tol_rare_label_compounds = trial.suggest_float('tol_rare_label_compounds',previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[0],previous_stage_param_range(study_1_params['tol_rare_label_compounds'],0.3)[1])
            mean_smoothing = trial.suggest_float('mean_smoothing',previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[0],previous_stage_param_range(study_1_params['mean_smoothing'],0.3)[1])
            feature_selector = trial.suggest_float('feature_selector',previous_stage_param_range(study_1_params['feature_selector'],0.3)[0],previous_stage_param_range(study_1_params['feature_selector'],0.3)[1])

            alpha = trial.suggest_float('alpha',previous_stage_param_range(study_1_params['alpha'],0.3,[0.001,1.0])[0],previous_stage_param_range(study_1_params['alpha'],0.3,[0.001,1.0])[1])
            l1_ratio = trial.suggest_float('l1_ratio',previous_stage_param_range(study_1_params['l1_ratio'],0.3,[0.001,1.0])[0],previous_stage_param_range(study_1_params['l1_ratio'],0.3,[0.001,1.0])[1])
            max_iter = trial.suggest_int('max_iter',previous_stage_param_range(study_1_params['max_iter'],0.3)[0],previous_stage_param_range(study_1_params['max_iter'],0.3)[1])
            tol = trial.suggest_float('tol',previous_stage_param_range(study_1_params['tol'],0.3)[0],previous_stage_param_range(study_1_params['tol'],0.3)[1])
            selection = study_1_params['selection']

            elastic_net_params = {
                'alpha':alpha,
                'l1_ratio':l1_ratio,
                'max_iter':max_iter,
                'tol':tol,
                'selection':selection,
                'random_state':101
            }

            #adding the fold_idx for the pruner, so the pruner knows, which trial we are
            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(smoke_X_train,smoke_y_train)):
                X_train_,y_train_ = smoke_X_train.iloc[train_idx],smoke_y_train.iloc[train_idx]
                X_val,y_val = smoke_X_train.iloc[val_idx],smoke_y_train.iloc[val_idx]

                #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
                #missing_in_sprint_we = [sprint_not for sprint_not in X_train_.columns if 'p2' in sprint_not or 'p3' in sprint_not]
                missing_input_features = [ip for ip in X_train_.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
                df_missing = X_train_[missing_input_features]
                missing_num = list(df_missing.select_dtypes(include=['number']).columns)
                missing_ob = list(df_missing.select_dtypes(exclude=['number']).columns)
                
                #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
                mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
                'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
                mean_enc_cols = [m+'_mean' for m in mean_freq]
                freq_enc_cols = [f+'_frequency' for f in mean_freq]

                one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

                cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

                #tol => percentage/frequency of appearance
                rf_pipeline = Pipeline([
                    ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=tol_rare_label_driver,replace_with='Rare_Driver')),
                    ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=tol_rare_label_gp,replace_with='Rare_GP')),
                    #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                    #because tree-based models can deal with NaN values
                    ("RareLabelEncoder_Compounds",RareLabelEncoder(
                        variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                        tol=tol_rare_label_compounds,
                        n_categories=2,
                        replace_with='Rare_Compound',
                        missing_values='ignore'
                    )),
                    ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
                    ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
                    ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
                    ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
                    ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
                    ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
                    ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
                    ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
                    ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
                    ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=mean_smoothing,unseen="encode")),
                    ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
                    ("Feature_Selector",RFPermutationRegressorSelector(threshold=feature_selector)),
                    ("RobustScaler",RobustScaler()),
                    ("Elastic_Net_Regressor",ElasticNet(**elastic_net_params))
                ]).fit(X_train_,y_train_)

                pred_val = rf_pipeline.predict(X_val)
                mae_fold = mean_absolute_error(y_val,pred_val)
                mae_fold_lst.append(mae_fold)
                running_mae = float(np.mean(mae_fold_lst))
                trial.report(running_mae, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mae_cv=np.mean(mae_fold_lst)
            return mae_cv

        study2 = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(n_startup_trials=5,seed=101),
            pruner=MedianPruner(n_startup_trials=10,n_warmup_steps=2,n_min_trials=2)
        )

        start_time = time.time()
        study2.optimize(stage2,n_trials=5,n_jobs=1,show_progress_bar=True)
        plt.figure(figsize=(12,6))
        plot_param_importances(study2)
        plt.title("Parameter Importance for Study 2")
        plt.show()

        plot_optimization_history(study2)
        plt.title("Optimization History for Study 2")
        plt.show()

        plot_timeline(study2)
        plt.title("Timeline for Study 2")
        plt.show()
        end_time = time.time()

        plt.rcdefaults()
        plt.style.use("default")

        study_2_params = study2.best_params
        study_2_params['selection'] = study_1_params['selection']
        study_2_best_trial = study2.best_value
        study_2_run_time = end_time - start_time
        print(f"Run time of study 2: {study_2_run_time}")

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_param = study_1_params
        else:
            print("Study 2 led to the best result")
            best_param = study_2_params

        if study_1_best_trial < study_2_best_trial:
            print("Study 1 led to the best result")
            best_score = study_1_best_trial
        else:
            print("Study 2 led to the best result")
            best_score = study_2_best_trial

        total_time = study_1_run_time + study_2_run_time

        joblib.dump(best_param,MODEL_CHECK_TRAIN/'elastic_net_best_param.joblib')
        joblib.dump(best_score,MODEL_CHECK_TRAIN/'elastic_net_best_score.joblib')
        joblib.dump(total_time,MODEL_CHECK_TRAIN/'elastic_net_run_time.joblib')

        missing_input_features = [ip for ip in X_train.columns if 'p1' in ip.lower() or 'p2' in ip.lower() or 'p3' in ip.lower() or 'sprint' in ip.lower()]
        df_missing = X_train[missing_input_features]
        missing_num = list(df_missing.select_dtypes(include=['number']).columns)
        missing_ob = list(df_missing.select_dtypes(include=['object','category']).columns)

        #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
        mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
        mean_enc_cols = [m+'_mean' for m in mean_freq]
        freq_enc_cols = [f+'_frequency' for f in mean_freq]

        one_hot_cols = ['Pace_profile','Type','Direction','Rule_Era']

        cat_cols_to_cast = list(dict.fromkeys(missing_ob + mean_enc_cols + freq_enc_cols + one_hot_cols))

        cv=GroupTimeSplit(
            group_column='gp_id',
            n_splits=3,
            test_group_size=3
        )

        elastic_net_params = {
                'alpha':best_param['alpha'],
                'l1_ratio':best_param['l1_ratio'],
                'max_iter':best_param['max_iter'],
                'tol':best_param['tol'],
                'selection':best_param['selection'],
                'random_state':101
            }

        final_hist_pipeline = Pipeline([
            ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=best_param['tol_rare_label_driver'],replace_with='Rare_Driver')),
            ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=best_param['tol_rare_label_gp'],replace_with='Rare_GP')),
            #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
            #because tree-based models can deal with NaN values
            ("RareLabelEncoder_Compounds",RareLabelEncoder(
                variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                tol=best_param['tol_rare_label_compounds'],
                n_categories=2,
                replace_with='Rare_Compound',
                missing_values='ignore'
            )),
            ('Adding_Columns_for_mean_&_frequency',Add_Column(col=mean_freq,name_= ['_mean','_frequency'])),
            ("Indicating_Missing_Value", AddMissingIndicator(missing_only=True,variables=missing_input_features)),
            ("NaN-Imputer_Numeric", ArbitraryNumberImputer(arbitrary_number=0,variables=missing_num)),
            ("Cast_Missing_Categorical_Before_Imputer", CastColumnsToObject(variables=missing_ob)),
            ("NaN-Imputer_Categorical", CategoricalImputer(imputation_method="missing",fill_value='-',variables=missing_ob)),
            ("Cast_Encoding_Cols_Before_Imputer", CastColumnsToObject(variables=mean_enc_cols + freq_enc_cols)),
            ("NaN_Imputer_Categorical_Encoding_Cols", CategoricalImputer(imputation_method="missing",fill_value="-",variables=mean_enc_cols + freq_enc_cols,)),
            ("Drop_Features",DropFeatures(features_to_drop=mean_freq)),
            ('One_Hot_Encoded_Features',OneHotEncoder(variables=one_hot_cols)),
            ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore',smoothing=best_param['mean_smoothing'],unseen="encode")),
            ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore',encoding_method='frequency',unseen="encode")),
            ("Feature_Selector",RFPermutationRegressorSelector(threshold=best_param['feature_selector'])),
            ("RobustScaler",RobustScaler()),
            ("Elastic_Net_Regressor",ElasticNet(**elastic_net_params))
        ]).fit(smoke_X_train,smoke_y_train)

        final_hist_pipeline.fit(smoke_X_train,smoke_y_train)
        y_pred = final_hist_pipeline.predict(smoke_X_test)
        elastic_net_mae = mean_absolute_error(smoke_y_test,y_pred)
        joblib.dump(elastic_net_mae,MODEL_CHECK_PRED/'prediction_smoke_test_elastic_net.joblib')
    except Exception as ex:
        logging.error(f"An error occurre while training, fitting and evaluating the Elastic Net model")
        raise

    logging.info("All smoke tests have succeeded")
