def retraining_final_model():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import cross_val_score,learning_curve,LearningCurveDisplay
    from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
    from feature_engine.encoding import MeanEncoder,OneHotEncoder,RareLabelEncoder,CountFrequencyEncoder
    from feature_engine.imputation import AddMissingIndicator,ArbitraryNumberImputer,CategoricalImputer
    from feature_engine.selection import DropFeatures
    from lightgbm import LGBMRegressor
    from optuna.visualization.matplotlib import plot_param_importances,plot_optimization_history,plot_timeline
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    import seaborn as sns
    import time
    import joblib
    import logging
    from pathlib import Path
    from .py_def_class import Add_Column,RFPermutationRegressorSelector,GroupTimeSplit,previous_stage_param_range,group_time_learning_curve,CastColumnsToObject,abandoned_lap,inspect_quali_time_distribution,f1_rule_era,get_sprint_session_name,build_session_paths,pick_quali_boundaries,practice_quali_new_pred,def_apply_offset_mean

    #setting up paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA = PROJECT_ROOT / 'data'
    DATA_PROCESSED = DATA/'processed'
    DATA_TRAIN_TEST = DATA/'model_data'
    DATA_FINAL = DATA_TRAIN_TEST/'final'
    DATA_TEST = DATA_TRAIN_TEST/'test'
    DATA_TRAIN = DATA_TRAIN_TEST/'train'
    #model_param
    MODEL_PARAMS = PROJECT_ROOT / 'model' / 'params'
    ##model
    MODEL = PROJECT_ROOT/'model'/'final_model'
    ##model_eval
    MODEL_EVAL = PROJECT_ROOT/'model'/'eval'
    ##model runtime
    MODEL_RUN_TIME = PROJECT_ROOT/'model'/'run_time'

    #setting up logging logic
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Loading the Training Data")
    try:
        X_train = pd.read_csv(DATA_TRAIN / 'X_train.csv')
        y_train = pd.read_csv(DATA_TRAIN / 'y_train.csv')['laptime_sum_sectortimes_quali']
    except Exception as ex:
        logging.error(f'An error occurred during loading the training data: {ex}')

    logging.info('Start retraining the model')
    try:
        def stage1(trial):
            cv=GroupTimeSplit(
                group_column='gp_id',
                n_splits=5,
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

            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(X_train,y_train)):
                X_train_,y_train_ = X_train.iloc[train_idx],y_train.iloc[train_idx]
                X_val,y_val = X_train.iloc[val_idx],y_train.iloc[val_idx]

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
        study1.optimize(stage1,n_trials=40,n_jobs=-1,show_progress_bar=True)
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
                n_splits=5,
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
            for fold_idx, (train_idx,val_idx) in enumerate(cv.split(X_train,y_train)):
                X_train_,y_train_ = X_train.iloc[train_idx],y_train.iloc[train_idx]
                X_val,y_val = X_train.iloc[val_idx],y_train.iloc[val_idx]

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
        study2.optimize(stage2,n_trials=60,n_jobs=1,show_progress_bar=True)
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
    except Exception as ex:
        logging.error(f'During Training an error occurred: {ex}')
        raise

    logging.info('Loading previous models performances')
    try:
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
            n_splits=5,
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
                'bagging_freq':best_param['bagging_freq'],
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

        final_hist_pipeline = rf_pipeline = Pipeline([
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
        ])

        train_group_sizes, train_mae, val_mae = (
            group_time_learning_curve(
                estimator=final_hist_pipeline,
                X=X_train,
                y=y_train,
                cv=cv,
                train_fractions=np.linspace(0.1, 1.0, 10),
            )
        )

        for size, fold_train_mae, fold_val_mae in zip(
            train_group_sizes,
            train_mae,
            val_mae,
        ):
            print(
                f"{size:.1f} GP groups | "
                f"Train MAE: {fold_train_mae.mean():.3f} "
                f"± {fold_train_mae.std():.3f} | "
                f"Validation MAE: {fold_val_mae.mean():.3f} "
                f"± {fold_val_mae.std():.3f} | "
                f"Gap: {fold_val_mae.mean() - fold_train_mae.mean():.3f}"
            )
    except Exception as ex:
        logging.error(f'An error occurred while loading')
        #this will stop the training process, if an error occurred
        raise

    try:
        df_performance = pd.DataFrame()

        #model_name
        model_name = ['DummyRegressor','HistGradientBoosting','LightGBM','XGBoosting','MLP','Lasso','Ridge','Elastic Net']

        #model scores
        dummy_score = joblib.load(MODEL_EVAL/'dummy_regression_performance.joblib')
        hist_score = joblib.load(MODEL_EVAL/'histgradientboosting_best_score.joblib')
        lgbm_score = joblib.load(MODEL_EVAL/'lightgbm_best_score.joblib')
        xgboost_score = joblib.load(MODEL_EVAL/'xgboost_best_score.joblib')
        mlp_score = joblib.load(MODEL_EVAL/'mlp_best_score.joblib')
        lasso_score = joblib.load(MODEL_EVAL/'lasso_best_score.joblib')
        ridge_score = joblib.load(MODEL_EVAL/'ridge_best_score.joblib')
        elastic_score = joblib.load(MODEL_EVAL/'elastic_net_best_score.joblib')
        model_score_lst = [dummy_score,hist_score,lgbm_score,xgboost_score,mlp_score,lasso_score,ridge_score,elastic_score]

        #run_time
        hist_run_time = (joblib.load(MODEL_RUN_TIME/'histgradientboosting_run_time.joblib')/60)/60
        lgbm_run_time = (joblib.load(MODEL_RUN_TIME/'lightgbm_run_time.joblib')/60)/60
        xgboost_run_time = (joblib.load(MODEL_RUN_TIME/'xgboost_run_time.joblib')/60)/60
        mlp_run_time = (joblib.load(MODEL_RUN_TIME/'mlp_run_time.joblib')/60)/60
        lasso_runt_time = (joblib.load(MODEL_RUN_TIME/'lasso_run_time.joblib')/60)/60
        ridge_runt_time = (joblib.load(MODEL_RUN_TIME/'ridge_run_time.joblib')/60)/60
        elastic_runt_time = (joblib.load(MODEL_RUN_TIME/'elastic_net_run_time.joblib')/60)/60
        model_run_time_lst = [((1/60)/60),hist_run_time,lgbm_run_time,xgboost_run_time,mlp_run_time,lasso_runt_time,ridge_runt_time,elastic_runt_time]

        df_performance = pd.DataFrame(
            {
                'Models':model_name,
                'Model_Score':model_score_lst,
                'Model_Run_Time_in_h':model_run_time_lst
            }
        ).sort_values('Model_Score')
        df_performance['score_rank'] = range(1,len(model_score_lst)+1)

        sub_df = df_performance[['Models','Model_Run_Time_in_h']].sort_values('Model_Run_Time_in_h')
        sub_df['run_time_rank'] = range(1,len(model_score_lst)+1)
        df_performance = df_performance.merge(sub_df,how='inner',on=['Models','Model_Run_Time_in_h'])
    except Exception as ex:
        logging.error(f'An error occurred while loading previous performances of the tuning of the models')
        raise

    best_prev_model = df_performance[df_performance['Model_Score'].eq(df_performance["Model_Score"].min())]
    logging.info(f'The best model was in the previous run: {best_prev_model["Models"].iloc[0]}')
    try:
        if best_score < float(best_prev_model['Model_Score'].iloc[0]):
            joblib.dump(best_param,MODEL_PARAMS/'lightgbm_best_param.joblib')
            joblib.dump(best_score,MODEL_EVAL/'lightgbm_best_score.joblib')
            joblib.dump(total_time,MODEL_RUN_TIME/'lightgbm_run_time.joblib')
            logging.info('The newly trained model has a better/lower score than the previous best model')
        else:
            logging.info('Previously best model is still the best model')
    except Exception as ex:
        logging.error(f'Something went wrong wile comparing models or saving parameters')
        raise

    logging.info('The retraining process is completed')

if __name__ == "__main__":
    retraining_final_model()