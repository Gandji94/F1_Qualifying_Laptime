def test_smoke_output_check():
    import pandas as pd
    import logging
    import pathlib

    logging.basicConfig(
            level = logging.INFO,
            format="%(asctime)s [%(levelname)s %(message)s]"
    )

    ROOT = pathlib.Path(__file__).resolve().parent.parent
    OUTPUT_TRAIN = ROOT/'model'/'check'/'train'
    OUTPUT_EVAL = ROOT/'model'/'check'/'prediction'

    logging.info("Looping through the training output")
    training_item_list=[]
    data_path = pathlib.Path(OUTPUT_TRAIN)
    for x in data_path.iterdir():
        training_item_list.append(x.name)

    models = [
        'histgradientboosting','lightgbm','xgboost',
        'lasso','ridge','elastic'
        ]
    for m in models:
        number_of_training_output_items = len([o for o in training_item_list if m in o])
        print(f"{m} - Number of training output items: {number_of_training_output_items}")
        assert number_of_training_output_items >= 3,f"{m} has less than 3 items"

    logging.info("Lopping through the evalaution results")
    pred_item_list=[]
    data_path = pathlib.Path(OUTPUT_EVAL)
    for x in data_path.iterdir():
        pred_item_list.append(x.name)
    model_pred = [x.split('_')[3].split('.')[0] for x in pred_item_list]
    for mp in model_pred:
        loop_sub_mp = [s for s in pred_item_list if mp in s]
        print(f"{mp} - Number of evaluation score items: {len(loop_sub_mp)}")
        assert len(loop_sub_mp) >= 1,f"{mp} has less than 1 evaluation score"