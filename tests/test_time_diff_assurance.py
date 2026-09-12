def test_time_difference():
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


    logging.info(f"LoadingX feature(s) sets")
    try:
        #we will import the all X and y features that we have available
        X_train = pd.read_csv(MODEL_TRAIN/'X_train.csv')
        X_test = pd.read_csv(MODEL_TEST/'X_test.csv')
        X_final = pd.read_csv(MODEL_FINAL/'final_X.csv')

        #we also include the the seperate the data drift features
        X_old = pd.read_csv(DRIFT/'X_features_old.csv')
        X_new = pd.read_csv(DRIFT/'X_new.csv')
    except Exception as ex:
        logging.error(f"An error occurred while loading the X feature(s) sets: {ex}")
        raise

    logging.info("Checking if there are sets which have a smaller max value than train max")
    train_max = X_train['Season'].max()
    test_max = X_test['Season'].max()
    final_max = X_final['Season'].max()

    old_max = X_old['Season'].max()
    new_max = X_new['Season'].max()

    #checking if any of the following sets have a smaller max value than the max of the training set
    for t,name in [(test_max,'test'),(final_max,'final'),(old_max,'old data drift'),(new_max,'new data drift')]:
        assert t >= train_max,f"{name} set has a smaller max value than training set"
    logging.info('Test was successful')