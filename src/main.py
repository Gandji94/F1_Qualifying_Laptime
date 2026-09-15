import argparse
from .all_model_training import all_model_training
from .eval_final_model_w_offset import eval_final_model_w_offset
from .eval_final_train_test_model import eval_final_train_test_model
from .fitting_final_model import fitting_final_model
from .archive.making_predictions import making_predictions
from .make_prediction_ import make_prediction_
#from .tests.pipeline_test import pipeline_test
from .quali_time_cleaning import quali_time_cleaning
from .retraining_final_model import retraining_final_model
from .train_test_split import train_test_split
from .data_drift import data_drift
from .feature_impact_on_pred_shap import feature_impact_on_pred_shap
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def parse_args():
    parser=argparse.ArgumentParser(
        description="Orchstrator for the F1 Qualifying Times Model."
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help = 'What function do you want to run?'
    )

    subparsers.add_parser(
        'quali_time_cleaning',
        help='Cleaning and merging the data.'
    )

    subparsers.add_parser(
        'train_test_split',
        help='Splitting data into train & test.'
    )

    #subparsers.add_parser(
        #'pipeline_test',
        #help='Checking if the pipeline is returning the wanted output.'
    #)

    subparsers.add_parser(
        'all_model_training',
        help='Here we can train all models all over again.'
    )

    subparsers.add_parser(
        'fitting_final_model',
        help = 'Fitting the final model with the test & train data (Note, extra hold out for offset).'
    )

    subparsers.add_parser(
        'eval_final_train_test_model',
        help = 'Checking performance of the final model when testing on hold out data.'
    )

    subparsers.add_parser(
        'eval_final_model_w_offset',
        help='Checking performance of the final model when testing on hold out data with the offset.'
    )

    subparsers.add_parser(
        'retraining_final_model',
        help='Retraining the final model, instead of training all, we can try to just retrain the final model, if there is one available.'
    )

    subparsers.add_parser(
        'make_prediction_',
        help='Making predictions with the final model'
    )

    subparsers.add_parser(
        'data_drift',
        help='Analysing the X features (raw and encoded), target feature and evalauation metrics'
    )

    subparsers.add_parser(
        "final_model_run",
        help="Retraining and fitting the current best model"
    )

    subparsers.add_parser(
        "full_run",
        help="Run the complete F1 model workflow."
    )

    subparsers.add_parser(
        "feature_impact_on_pred_shap",
        help="Anaylsing the impact of the features on the prediction"
    )

    return parser.parse_args()

def main():
    args=parse_args()
    logging.info(f'[DEBUG] command = {args.command}')

    if args.command == 'quali_time_cleaning':
        quali_time_cleaning()
    elif args.command == 'train_test_split':
        train_test_split()
    #elif args.command == 'pipeline_test':
        #pipeline_test()
    elif args.command == 'all_model_training':
        all_model_training()
    elif args.command == 'fitting_final_model':
        fitting_final_model()
    elif args.command == 'eval_final_train_test_model':
        eval_final_train_test_model()
    elif args.command == 'eval_final_model_w_offset':
        eval_final_model_w_offset()
    elif args.command == 'retraining_final_model':
        retraining_final_model()
    elif args.command == 'make_prediction_':
        make_prediction_(df_filter=None)
    elif args.command == 'data_drift':
        data_drift(season_ahead=2)
    elif args.command == 'feature_impact_on_pred_shap':
        feature_impact_on_pred_shap()
    elif args.command == "final_model_run":
        quali_time_cleaning()
        train_test_split()
        retraining_final_model()
        fitting_final_model()
        eval_final_train_test_model()
        eval_final_model_w_offset()
        making_predictions(df_filter=None)
        make_prediction_()
    elif args.command == "full_run":
        quali_time_cleaning()
        train_test_split()
        all_model_training()
        fitting_final_model()
        eval_final_train_test_model()
        eval_final_model_w_offset()
        making_predictions(df_filter=None)
        make_prediction_()

if __name__ == '__main__':
    main()

#how to call the functions
##python -m src.main quali_time_cleaning
##python -m src.main train_test_split
##python -m src.main pipeline_test
##python -m src.main all_model_training
##python -m src.main fitting_final_model
##python -m src.main eval_final_train_test_model
##python -m src.main eval_final_model_w_offset
##python -m src.main retraining_final_model
##python -m src.main making_prediction_
##python -m src.main data_drift
##python -m src.main final_model_run
##python -m src.main full_run