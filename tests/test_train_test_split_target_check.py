def test_train_test_split_check():
    import pandas as pd 
    import numpy as np
    import logging
    import joblib
    import pathlib


    logging.basicConfig(
        level = logging.INFO,
        format="%(asctime)s [%(levelname)s %(message)s]"
    )


    ROOT = pathlib.Path(__file__).resolve().parent.parent
    DATA = ROOT/'data'
    MODEL = DATA/'model_data'
    MODEL_TRAIN = MODEL/'train'
    MODEL_TEST = MODEL/'test'
    MODEL_FINAL = MODEL/'final'
    DRIFT = DATA/'data_drift'


    logging.info(f"Loading the y and X feature(s) sets")
    try:
        #we will import the all X and y features that we have available
        y_train = pd.read_csv(MODEL_TRAIN/'y_train.csv')
        y_test = pd.read_csv(MODEL_TEST/'y_test.csv')
        y_final = pd.read_csv(MODEL_FINAL/'final_y.csv')

        X_train = pd.read_csv(MODEL_TRAIN/'X_train.csv')
        X_test = pd.read_csv(MODEL_TEST/'X_test.csv')
        X_final = pd.read_csv(MODEL_FINAL/'final_X.csv')

        #we also include the the seperate the data drift features
        y_old = pd.read_csv(DRIFT/'target_feature.csv')
        y_new = pd.read_csv(DRIFT/'y_new.csv')

        X_old = pd.read_csv(DRIFT/'X_features_old.csv')
        X_new = pd.read_csv(DRIFT/'X_new.csv')
    except Exception as ex:
        logging.error(f"An error occurred while loading the y and X feature(s) sets: {ex}")
        raise

    logging.info("Checking if X and y sets indexes match")
    for y,x,name in [(y_train,X_train,'train'),(y_test,X_test,'test'),(y_final,X_final,'final'),(y_old,X_old,'data drift old'),(y_new,X_new,'data drift new')]:
        #checking if the indexes between the X and y equivilants are matching
        assert y.index.equals(x.index), f"{name}: DataFrames have different indexes"

    logging.info("Testing for remaining NaN values in the target feature sets")

    for y,x,name in [(y_train,X_train,'train'),(y_test,X_test,'test'),(y_final,X_final,'final'),(y_old,X_old,'data drift old'),(y_new,X_new,'data drift new')]:
        #ckecking for NaN values in the target variables
        ## we use any().any, because the y feature/target is saved as data frame
        assert not y.isna().any().any(), f"{y}: Contains NaN values"

    logging.info('All tests were successful')