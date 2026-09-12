def feature_impact_on_pred_shap():
    import pandas as pd
    import numpy as np
    import joblib
    import shap
    import pathlib
    import logging
    import src.py_def_class

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    DATA = PROJECT_ROOT / 'data'
    DATA_TRAIN_TEST = DATA/'model_data'
    DATA_TEST = DATA_TRAIN_TEST/'test'
    DATA_TRAIN = DATA_TRAIN_TEST/'train'
    MODEL = PROJECT_ROOT/'model'/'final_model'
    MODEL_SHAP = PROJECT_ROOT/'model'/'shap'

        #setting up logging logic
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Setting up the SAHP MAE for every Feature")
    try:
        final_model = joblib.load(MODEL/"final_model_qualifying.joblib")

        X_test = pd.read_csv(DATA_TEST/"X_test.csv")
        X_train = pd.read_csv(DATA_TRAIN/"X_train.csv")
        y_test = pd.read_csv(DATA_TEST/"y_test.csv")

        step_names = list(final_model.named_steps.keys())
        selector_idx = step_names.index("Feature_Selector")
        #everything BEFORE Feature_Selector
        X_train_encoded = final_model[:selector_idx].transform(X_train)
        X_test_encoded = final_model[:selector_idx].transform(X_test)
        # Feature selector
        selector = final_model["Feature_Selector"]

        #all the selected features
        X_train_selected = selector.transform(X_train_encoded)
        X_test_selected = selector.transform(X_test_encoded)

        print(X_test_selected.columns)

        model_after_selector = final_model[selector_idx + 1:]
        background = shap.sample(
            X_train_selected,
            100,
            random_state=42
        )

        explainer = shap.Explainer(
            model_after_selector.predict,
            background
        )

        shap_values = explainer(X_test_selected)

        global_shap = pd.DataFrame({
            "Feature": X_test_selected.columns,
            "Mean_ABS_SHAP": np.abs(shap_values.values).mean(axis=0)
        })

        global_shap = (
            global_shap
            .sort_values("Mean_ABS_SHAP", ascending=False)
            .reset_index(drop=True)
        )
        global_shap["Importance_%"] = (
            global_shap["Mean_ABS_SHAP"]
            / global_shap["Mean_ABS_SHAP"].sum()
            * 100
        )
        global_shap.to_csv(MODEL_SHAP/"feature_impact_on_pred_shap.csv",index=False)
        print(global_shap)
    except Exception as ex:
        logging.error(f"An error occurred while setting up the SAHP MAE for every Feature")

    logging.info("Completed")