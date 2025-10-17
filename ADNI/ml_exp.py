import os
import logging
from flaml import AutoML
import joblib
from datetime import datetime
import pandas as pd

import sys
sys.path.append('../ukb')
from doubleml_utils import subset_train_test

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument("--experiment", type=str, required=True, help = 
                        "age_alone: age as the only feature" \
                        "apoe_alone: apoe as the only feature" \
                        "all: age, apoe, and lancets as features")

    args = parser.parse_args()

    fold_index = args.fold_index
    experiment = args.experiment

    return fold_index, experiment

def settings_automl(time_budget, metric):
    """
    Generate settings for an AutoML classification task.
    Parameters:
    time_budget (int): The time budget for the AutoML process in seconds.
    metric (str): The evaluation metric to be used (e.g., 'log_loss' 'accuracy', 'f1').
    model (str): The model to be used in the AutoML process (e.g., 'lrl1').
    region_index (int): The index of the region for logging purposes.
    Returns:
    dict: A dictionary containing the settings for the AutoML process.
    """
    automl_settings = {
            "task": "classification",
            "time_budget": time_budget,
            "metric": metric,
            "n_jobs": -1,
            "eval_method": "cv",
            "n_splits": 5,
            "early_stop": True,
            "log_training_metric": True,
            "model_history": True,
            "seed": 1234321,
            "estimator_list": ['lgbm'],
    }

    return automl_settings

def subset_experiment_vars(df, experiment): 
    if experiment == 'age':
        df = df[['AGE', 'DXAD']]
        return df
    elif experiment == 'apoe':
        df = df[['apoe_2/2', 'apoe_2/3', 'apoe_2/4', 'apoe_3/3', 'apoe_3/4', 'apoe_4/4', 'DXAD']]
        return df
    elif experiment == 'all':
        return df

def main(): 
    fold_index, experiment = parse_args()

    df = pd.read_parquet('ADNI_cleaned.parquet', engine='fastparquet')
    df = subset_experiment_vars(df, experiment)

    X = df.drop(columns=['DXAD'])
    y = df['DXAD']
    
    print('splitting')
    
    results_dir = f'./flaml_results/{experiment}/{fold_index}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    X_train, y_train, X_test, y_test = subset_train_test(X, y, results_dir, fold_index)

    print(f'training fold {fold_index} with experiment {experiment}')
    automl = AutoML()
    automl_settings = settings_automl(300, metric="log_loss")
    print(automl_settings)

    logging.info(f"Saving the model: {datetime.now().time()}")

    automl.fit(X_train, y_train, **automl_settings)

    # save the model
    best_model = automl.model.estimator

    # Save just the best model
    joblib.dump(best_model, f"{results_dir}/flaml_best_model.joblib")

    logging.info(f"Saving the predictions: {datetime.now().time()}")
    # save the test set predictions
    y_pred = automl.predict_proba(X_test)
    results = pd.DataFrame({"y_test": y_test, "y_pred": y_pred[:,1]})
    results.to_parquet(
        f"{results_dir}/test_labels_predictions.parquet", index=False, engine = 'fastparquet'
    )

    # save the train set predictions
    y_pred = automl.predict_proba(X_train)
    results = pd.DataFrame({"y_train": y_train, "y_pred": y_pred[:,1]})
    results.to_parquet(
        f"{results_dir}/train_labels_predictions.parquet", index=False, engine = 'fastparquet'
    )

if __name__ == "__main__":
    main()