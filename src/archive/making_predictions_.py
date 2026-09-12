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
from ..py_def_class import Add_Column,RFPermutationRegressorSelector,GroupTimeSplit,previous_stage_param_range,group_time_learning_curve,CastColumnsToObject,abandoned_lap,inspect_quali_time_distribution,f1_rule_era,get_sprint_session_name,build_session_paths,pick_quali_boundaries,practice_quali_new_pred,def_apply_offset_mean,tune_offset_shrinkage_expanding

#setting up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT/'data'
DATA_RAW = DATA/'raw'
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

logging.info('Setting up the F1 Race Calendars')
#setting up the F1 season claendar
f1_calendar_by_season = {
    2018: [
        "Australian GP", "Bahrain GP", "Chinese GP", "Azerbaijan GP",
        "Spanish GP", "Monaco GP", "Canadian GP", "French GP",
        "Austrian GP", "British GP", "German GP", "Hungarian GP",
        "Belgian GP", "Italian GP", "Singapore GP", "Russian GP",
        "Japanese GP", "United States GP", "Mexican GP", "Brazilian GP",
        "Abu Dhabi GP",
    ],

    2019: [
        "Australian GP", "Bahrain GP", "Chinese GP", "Azerbaijan GP",
        "Spanish GP", "Monaco GP", "Canadian GP", "French GP",
        "Austrian GP", "British GP", "German GP", "Hungarian GP",
        "Belgian GP", "Italian GP", "Singapore GP", "Russian GP",
        "Japanese GP", "Mexican GP", "United States GP", "Brazilian GP",
        "Abu Dhabi GP",
    ],

    2020: [
        "Austrian GP", "Styrian GP", "Hungarian GP", "British GP",
        "70th Anniversary GP", "Spanish GP", "Belgian GP", "Italian GP",
        "Tuscan GP", "Russian GP", "Eifel GP", "Portuguese GP",
        "Emilia Romagna GP", "Turkish GP", "Bahrain GP", "Sakhir GP",
        "Abu Dhabi GP",
    ],

    2021: [
        "Bahrain GP", "Emilia Romagna GP", "Portuguese GP", "Spanish GP",
        "Monaco GP", "Azerbaijan GP", "French GP", "Styrian GP",
        "Austrian GP", "British GP", "Hungarian GP", "Belgian GP",
        "Dutch GP", "Italian GP", "Russian GP", "Turkish GP",
        "United States GP", "Mexican GP", "Brazilian GP", "Qatar GP",
        "Saudi Arabian GP", "Abu Dhabi GP",
    ],

    2022: [
        "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Emilia Romagna GP",
        "Miami GP", "Spanish GP", "Monaco GP", "Azerbaijan GP",
        "Canadian GP", "British GP", "Austrian GP", "French GP",
        "Hungarian GP", "Belgian GP", "Dutch GP", "Italian GP",
        "Singapore GP", "Japanese GP", "United States GP", "Mexican GP",
        "Brazilian GP", "Abu Dhabi GP",
    ],

    2023: [
        "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Azerbaijan GP",
        "Miami GP", "Monaco GP", "Spanish GP", "Canadian GP",
        "Austrian GP", "British GP", "Hungarian GP", "Belgian GP",
        "Dutch GP", "Italian GP", "Singapore GP", "Japanese GP",
        "Qatar GP", "United States GP", "Mexican GP", "Brazilian GP",
        "Las Vegas GP", "Abu Dhabi GP",
    ],

    2024: [
        "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Japanese GP",
        "Chinese GP", "Miami GP", "Emilia Romagna GP", "Monaco GP",
        "Canadian GP", "Spanish GP", "Austrian GP", "British GP",
        "Hungarian GP", "Belgian GP", "Dutch GP", "Italian GP",
        "Azerbaijan GP", "Singapore GP", "United States GP", "Mexican GP",
        "Brazilian GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP",
    ],

    2025: [
        "Australian GP", "Chinese GP", "Japanese GP", "Bahrain GP",
        "Saudi Arabian GP", "Miami GP", "Emilia Romagna GP", "Monaco GP",
        "Spanish GP", "Canadian GP", "Austrian GP", "British GP",
        "Belgian GP", "Hungarian GP", "Dutch GP", "Italian GP",
        "Azerbaijan GP", "Singapore GP", "United States GP", "Mexican GP",
        "Brazilian GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP",
    ],

    #2026 was already adjusted due to the canceld Saudi Arabia and Barhain GPs
    2026: [
        "Australian GP", "Chinese GP", "Japanese GP", "Miami GP",
        "Canadian GP", "Monaco GP", "Barcelona-Catalunya GP", "Austrian GP",
        "British GP", "Belgian GP", "Hungarian GP", "Dutch GP",
        "Italian GP", "Spanish GP", "Azerbaijan GP", "Singapore GP",
        "United States GP", "Mexican GP", "Brazilian GP", "Las Vegas GP",
        "Qatar GP", "Abu Dhabi GP",
    ],
}

logging.info('Loading the circut information')
try:
    #loading the circut information data frame
    df_circut = pd.read_csv(DATA_PROCESSED / 'cleaned_f1_circut.csv')
except Exception as ex:
    logging.error(f'An error occurred while loading the circut information: {ex}')
    raise

logging.info('Creating GP and session information')
try:
    df_2018_2026 = pd.DataFrame()
    start_range = 2018
    end_range = 2026

    for season in range(start_range,end_range+1):
        #turning "season" to string, because the path cannot processes a numerical value
        data_path = DATA_RAW / str(season)

        #getting name of folder if the folder exists
        path_list = sorted([f.name for f in data_path.iterdir() if f.is_dir() and not f.name.startswith('.') and f.name.endswith("GP")])

        # keep this only if you really need it
        #if season <= 2026:
        #path_list = path_list[1:]

        if season == 2020:
            path_list = [x for x in path_list if x != "Eifel GP"]

        #entering each folder of a Grand Prix weekend
        for folder_loop_value in path_list:
            base = data_path / folder_loop_value

            timing_paths, weather_paths = build_session_paths(
                base=base,
                season=season,
                folder_loop_value=folder_loop_value
            )

            df_loop = practice_quali_new_pred(
                season=season,
                gp=folder_loop_value,
                timing_paths=timing_paths,
                weather_paths=weather_paths,
                w_vis=False
            )

            df_2018_2026 = pd.concat([df_2018_2026, df_loop], ignore_index=True)
    #creating the mapping => key: season, gp; value: number of the gp-weekend
    race_round_map = {
        (season, gp): round_no
        for season, races in f1_calendar_by_season.items()
        for round_no, gp in enumerate(races, start=1)
    }

    #race_round will be used as an X-feature
    df_2018_2026["race_round"] = df_2018_2026.apply(
        lambda row: race_round_map.get((row["Season"], row["GP"])),
        axis=1
    )

    #creating a dataframe to create an race order
    weekend_order = (
        df_2018_2026[["Season", "GP", "race_round"]]
        .drop_duplicates()
        .sort_values(["Season", "race_round"])
        .reset_index(drop=True)
    )
    #creating a unique id based on the race weekend order
    #group_id will be used for time-aware cross-validation
    weekend_order["gp_id"] = range(len(weekend_order))

    #finally joining the race order logic
    df_2018_2026 = df_2018_2026.merge(
        weekend_order[["Season", "GP", "gp_id"]],
        on=["Season", "GP"],
        how="left"
    )

    team_lineage_map = {
        "Toro Rosso": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
        "AlphaTauri": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
        "RB": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
        "Racing Bulls": "Toro Rosso-AlphaTauri-RB-Racing Bulls",

        "Sauber": "Sauber-Alfa Romeo-Kick Sauber-Audi",
        "Alfa Romeo": "Sauber-Alfa Romeo-Kick Sauber-Audi",
        "Alfa Romeo Racing": "Sauber-Alfa Romeo-Kick Sauber-Audi",
        "Kick Sauber": "Sauber-Alfa Romeo-Kick Sauber-Audi",
        "Audi": "Sauber-Alfa Romeo-Kick Sauber-Audi",

        "Force India":"Force India-Racing Point-Aston Martin",
        "Racing Point":"Force India-Racing Point-Aston Martin",
        "Aston Martin":"Force India-Racing Point-Aston Martin",
        "Alpine":"Alpine-Renault",
        "Renault":"Alpine-Renault"
    }

    df_2018_2026["Team_Lineage"] = df_2018_2026["Team"].replace(team_lineage_map)
    df_quali = pd.merge(df_2018_2026,df_circut,how='left',on='GP')
    df_quali["Pace_profile"] = df_quali["Pace_profile"].str.strip()
    current_season = df_quali['Season'].max()
    df_quali = df_quali[df_quali['Season'].eq(current_season)]
except Exception as ex:
    logging.error(f'An error occurred while generating GP and session information: {ex}')
    raise

#---------------------------------------------- This block is for picking a GP ----------------------------------------------
#determining the Grand Prix
gp_f = str(input('Please enter the GP that you want to predict'))
#using the most recent data for teh Grand Prix
current_season = df_quali['Season'].max()
df_quali = df_quali[(df_quali['GP'].eq(f'{gp_f} GP')) & (df_quali['Season'].eq(current_season))].copy()

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
    null_check_df = df_quali.isnull().sum().to_frame()
    null_lst = set(n for n in null_check_df[null_check_df[0]>10].index)
    total_set_check = set(d for d in df_quali.columns if d.startswith(('AirTemp_','Humidity_','Pressure_','Rainfall_')))
    mean_valid_cols = total_set_check - null_lst

    for a in add_feat:
        sub_mean_cols = [smc for smc in mean_valid_cols if smc.split('_')[0] == a.split('_')[0]]
        print(sub_mean_cols)
        input_add_feat = float(np.mean([df_quali[x].mean() for x in sub_mean_cols]))
        df_quali[a]= input_add_feat
    df_quali['Compound_quali'] = input("Please enter Compound: ").strip().upper()
    df_quali = df_quali[[x for x in X_train.columns]].copy()
    #dropping the track temperature columns
    filter_out = set(fo for fo in df_quali.columns if 'TrackTemp' in fo)
    df_quali = df_quali.drop(filter_out,axis=1)
    df_quali = df_quali[df_quali['laptime_sum_sectortimes_quali'].notna()].copy()
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
    df_quali = def_apply_offset_mean(df=df_quali,pred='Pred_time',shrinking=0.55)
    prediction_output = (df_quali[["Driver", "Team", "GP", "pred_corrected"]].sort_values(["GP", "pred_corrected"],ascending=[True, True],).reset_index(drop=True))
    prediction_output.to_csv(MODEL_PRED/'2026_qualifying_predictions.csv',index=False)
except Exception as ex:
    logging.error(f'An error occurred while making offset predictions: {ex}')
    raise
logging.info('Completed')
print(prediction_output)