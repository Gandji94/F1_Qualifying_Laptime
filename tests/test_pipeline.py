def test_pipeline():
    #importing libraries
    import pandas as pd
    import numpy as np
    from feature_engine.encoding import MeanEncoder,OneHotEncoder,RareLabelEncoder,CountFrequencyEncoder
    from feature_engine.imputation import AddMissingIndicator
    from feature_engine.selection import DropFeatures
    from sklearn.base import BaseEstimator,TransformerMixin
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.utils.validation import check_is_fitted
    from sklearn.pipeline import Pipeline
    from src.py_def_class import Add_Column,RFPermutationRegressorSelector
    import pathlib
    import logging

    #setting up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    #setting up the paths
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    DATA = ROOT/'data'
    MODEL = DATA/'model_data'
    MODEL_TRAIN = MODEL/'train'
    MODEL_TEST = MODEL/'test'
    MODEL_FINAL = MODEL/'final'

    logging.info("Loading data")
    try:
        #we will import the all X and y features that we have available
        y_train = pd.read_csv(MODEL_TRAIN/'y_train.csv').iloc[:,0]
        y_test = pd.read_csv(MODEL_TEST/'y_test.csv').iloc[:,0]
        y_final = pd.read_csv(MODEL_FINAL/'final_y.csv').iloc[:,0]

        X_train = pd.read_csv(MODEL_TRAIN/'X_train.csv')
        X_test = pd.read_csv(MODEL_TEST/'X_test.csv')
        X_final = pd.read_csv(MODEL_FINAL/'final_X.csv')
    except Exception as ex:
        logging.error(f"An error occurred while loading the data: {ex}")
        raise

    logging.info("Setting up the pipeline")
    try:
        for y,X,name in [(y_train,X_train,'train'),(y_test,X_test,'test'),(y_final,X_final,'final')]:
            #list with all p2 and p3 features, which will be NaN, when we have a Sprint-Weekend
            missing_in_sprint_we = [sprint_not for sprint_not in X.columns if 'p2' in sprint_not or 'p3' in sprint_not]
            #
            mean_freq = ['Driver','GP','Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali',
            'Team','Team_Lineage','Complexity_label','Sprint-Session','Sprint_Race_Era']
            mean_enc_cols = [m+'_mean' for m in mean_freq]
            freq_enc_cols = [f+'_frequency' for f in mean_freq]

            #tol => percentage/frequency of appearance
            test_pipeline = Pipeline([
                ("RareLabel_Driver",RareLabelEncoder(variables='Driver',tol=0.0019,replace_with='Rare_Driver')),
                ("RareLabelEncoder_GP",RareLabelEncoder(variables='GP',tol=0.01,replace_with='Rare_GP')),
                #due to sprint weekends, some rows for the practice 2 and 3 compound tyre features include NaN, because we set missing_values='ignore'
                #because tree-based models can deal with NaN values
                ("RareLabelEncoder_Compounds",RareLabelEncoder(
                    variables=['Compound_p1','Compound_p2','Compound_p3','Compound_sprint_quali','Compound_quali'],
                    tol=0.015,
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
                ("Mean_encoding",MeanEncoder(variables=mean_enc_cols,missing_values='ignore')),
                ("Frequency_encoding",CountFrequencyEncoder(variables=freq_enc_cols,missing_values='ignore')),
                ("Feature_Selector",RFPermutationRegressorSelector(threshold=0.1))
            ]).fit_transform(X,y)

            try:
                logging.info(f"Pipeline output shape for {name}: {test_pipeline.shape}")

                object_cols = test_pipeline.select_dtypes(include="O").columns.tolist()
                nan_cols = test_pipeline.columns[test_pipeline.isna().any()].tolist()

                assert test_pipeline.shape[0] > 0, (
                    f"{name} Pipeline output has zero rows."
                )

                assert test_pipeline.shape[1] > 0, (
                    f"{name} Pipeline output has zero columns. "
                    "Feature selector probably selected no features."
                )

                assert len(object_cols) == 0, (
                    f"{name} Pipeline still contains object columns: "
                    f"{object_cols}"
                )

                logging.info(
                    f"{name} Pipeline data transformation was a success"
                )

                if len(nan_cols) > 0:
                    logging.warning(
                        f"{name} Pipeline still contains NaN values "
                        f"in columns: {nan_cols}"
                    )

            except Exception as ex:
                logging.error(f"Error occurred, pipeline does not match expectations {name}: {ex}")
                raise
    except Exception as ex:
            logging.error(f"Error while running the pipeline: {ex}")
            raise
    logging.info("Test was successful")