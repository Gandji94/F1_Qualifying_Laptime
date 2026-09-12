def fitting_final_model():
    #importing libaries
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
    import logging
    import joblib
    from pathlib import Path
    import sys
    #sys.path.append('../src')
    from .py_def_class import Add_Column,RFPermutationRegressorSelector,GroupTimeSplit,previous_stage_param_range,group_time_learning_curve,CastColumnsToObject,abandoned_lap,inspect_quali_time_distribution,f1_rule_era,get_sprint_session_name,build_session_paths,pick_quali_boundaries,practice_quali_new_pred,def_apply_offset_mean

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

    #setting up logging logic
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info('Loading the total train and test data for fitting')

    try:
        #seeting up logging info
        X = pd.read_csv(DATA_FINAL/'final_X.csv')
        y = pd.read_csv(DATA_FINAL/'final_y.csv')['laptime_sum_sectortimes_quali']
    except Exception as ex:
        logging.error(f"An error occurred loading the final X and y feature(s): {ex}")

    logging.info('Fitting the final model')

    try:
        #fitting the final model
        best_param = joblib.load(MODEL_PARAMS/'lightgbm_best_param.joblib')

        #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
        missing_in_sprint_we = [sprint_not for sprint_not in X.columns if 'p2' in sprint_not or 'p3' in sprint_not]
        #
        #mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        #'Team','Team_Lineage','Complexity_label','Session','Sprint-Session','Sprint_Race_Era']
        mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
        'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
        mean_enc_cols = [m+'_mean' for m in mean_freq]
        freq_enc_cols = [f+'_frequency' for f in mean_freq]

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

        final_pipeline = rf_pipeline = Pipeline([
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
            ("Light_GBM",LGBMRegressor(**lightgbm_params))
        ]).fit(X,y)

        joblib.dump(final_pipeline,MODEL/'final_model_qualifying.joblib')
    except Exception as ex:
        logging.error(f'An error occurred while fitting the final model: {ex}')

    logging.info('Successfull fitting')

if __name__ == "__main__":
    fitting_final_model()