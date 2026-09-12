def train_test_split():
    #importing libaries
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import logging

    #setting up paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA = PROJECT_ROOT / 'data'
    DATA_PROCESSED = DATA/'processed'
    DATA_TRAIN_TEST = DATA/'model_data'
    DATA_FINAL = DATA_TRAIN_TEST/'final'
    DATA_TEST = DATA_TRAIN_TEST/'test'
    DATA_TRAIN = DATA_TRAIN_TEST/'train'
    DATA_DRIFT = DATA/'data_drift'

    #setting up logging logic
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info('Loading Data')
    try:
        df = pd.read_csv(DATA_PROCESSED/'Cleaned_quali_f1.csv')
        #dropping the track temperature columns
        filter_out = set(fo for fo in df.columns if 'TrackTemp' in fo)
        #adding Qualifying_Date to the set that will be filtered out
        filter_out.add('Qualifying_Date')
        df = df.drop(filter_out,axis=1)
        df = df[df['laptime_sum_sectortimes_quali'].notna()].copy()
    except Exception as ex:
        logging.error(f'A problem occurred while loading the data: {ex}')
        raise

    logging.info('Splitting the data')
    try:
        target = "laptime_sum_sectortimes_quali"
        cols_not_in_X = set(filter_out) | {target}
        current_season = df['Season'].max()
        logging.info(f'Current Season: {current_season}')
        train_thresh = current_season - 2
        logging.info(f'Max Training Season: {train_thresh}')
        test_thresh = current_season - 1
        logging.info(f'Max Testing Season: {test_thresh}')

        X_train = df[df['Season']<=train_thresh].drop([target,'LapTimeDiff_quali','Session'],axis=1)
        X_train.to_csv(DATA_TRAIN/'X_train.csv',index=False)

        X_test = df[df['Season'].eq(test_thresh)].drop([target,'LapTimeDiff_quali','Session'],axis=1)
        X_test.to_csv(DATA_TEST/'X_test.csv',index=False)

        #-------------- checking for any leak or logic errors in X features --------------
        #checking if we do not have any overlapping indexies
        assert set(X_train.index).isdisjoint(set(X_test.index)),"Having overlapping indexies in the X feature-set"
        #checking if features that should be dropped are still in the train and/or test sets
        assert set(X_train.columns).isdisjoint(cols_not_in_X),"Having dropped features are still in the Train X features"
        assert set(X_test.columns).isdisjoint(cols_not_in_X),"Having dropped features are still in the Test X features"
        assert X_train['Season'].max() < X_test['Season'].min(),"Max Season value of X train is not smaller than min X test Season value"

        y_train = df[df['Season']<=train_thresh][target]
        y_train.to_csv(DATA_TRAIN/'y_train.csv',index=False)
        y_test = df[df['Season'].eq(test_thresh)][target]
        y_test.to_csv(DATA_TEST/'y_test.csv',index=False)
        #-------------- checking for any logic errors in y feature --------------
        assert set(y_train.index).isdisjoint(set(y_test.index)),"Having overlapping indexies in the y feature-set"
    except Exception as ex:
        logging.error(f"An error occurred while splitting the data: {ex}")
        raise
    logging.info('Combining and saving the train and test data')
    try:
        X = pd.concat([X_train,X_test],axis=0)
        X.to_csv(DATA_FINAL/'final_X.csv',index=False)
        X.to_csv(DATA_DRIFT/'X_features_old.csv',index=False)

        y = pd.concat([y_train,y_test],axis=0)
        y.to_csv(DATA_FINAL/'final_y.csv',index=False)
        y.to_csv(DATA_DRIFT/'target_feature.csv',index=False)
    except Exception as ex:
        logging.error(f"An error occurred while creating the final model data: {ex}")
        raise

    logging.info('Splitting the new data')
    try:
        X_new = df[df['Season'].eq(current_season)].drop([target,'LapTimeDiff_quali','Session'],axis=1)
        X_new.to_csv(DATA_DRIFT/'X_new.csv',index=False)

        y_new = df[df['Season'].eq(current_season)][target]
        y_new.to_csv(DATA_DRIFT/'y_new.csv',index=False)
    except Exception as ex:
        logging.error(f'An error occurred while splitting the new data: {ex}')
    logging.info('The train-test split is completed')

if __name__ == "__main__":
    train_test_split()