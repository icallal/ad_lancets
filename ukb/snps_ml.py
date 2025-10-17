'''
Alzheimer's patients vs. controls. Dementia patients have been excluded. 
'''
import pandas as pd
import numpy as np
import dementia_utils as dem

import argparse
import logging
import os
import sys
import joblib
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from flaml import AutoML

FLAML_TIME_BUDGET = 3600

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument(
        "--experiment",
        type=str,
        help="""Experiment name. Options:
        'proteins_alone' - use only the proteins as features
        'asp' - age, sex, proteins
        'proteins_omicspred' - proteins and omicspred scores
        'fs_proteins_alone' - using the results from 'proteins_alone', iteratively retrain the model with the top 1-100 features from the previous model
        'fs_asp' - using the results from 'asp', iteratively retrain the model with the top 1-100 features from the previous model
        """,
        required = False
    )
    return parser.parse_args()

# df = pd.read_parquet('raw_apoe_AD.parquet', engine = 'fastparquet') # apoe only
df = pd.read_parquet('../../../randy/proj_idp/tidy_data/prs_Alz/prs_Alz.parquet', engine = 'fastparquet')
if 'eid' in df.columns:
    df = df.rename(columns={'eid': 'IID'}) # polygenic risk score

# ddf = pd.read_parquet('65up_sex.parquet', engine = 'fastparquet') # age, sex, 65up
ddf = pd.read_parquet('../../../randy/proj_idp/tidy_data/acd/allcausedementia.parquet', engine = 'fastparquet') 

# df = df[df['IID'].isin(ddf['eid'])] # subset to over 65, comment if all ages

# ADDING ADDITIONAL FEATURES
# add age and sex

df = df.merge(ddf[['eid','curr_age', '31-0.0']], right_on = 'eid', left_on = 'IID')
df = df.drop(columns=['eid'])
df = df.rename(columns={
    'curr_age': 'age',
    '31-0.0': 'sex'
})


# encode APOE
df["e3/e3"] = 0
df["e3/e4"] = 0
df["e2/e3"] = 0
df["e2/e4"] = 0
df["e4/e4"] = 0
df["e2/e2"] = 0

df.loc[
    (df.rs429358_T == 2) & (df.rs7412_C == 2), "e3/e3"
] = 1
df.loc[
    (df.rs429358_T == 1) & (df.rs7412_C == 2), "e3/e4"
] = 1
df.loc[
    (df.rs429358_T == 2) & (df.rs7412_C == 1), "e2/e3"
] = 1
df.loc[
    (df.rs429358_T == 1) & (df.rs7412_C == 1), "e2/e4"
] = 1
df.loc[
    (df.rs429358_T == 0) & (df.rs7412_C == 2), "e4/e4"
] = 1
df.loc[
    (df.rs429358_T == 2) & (df.rs7412_C == 0), "e2/e2"
] = 1

df = df.drop(columns = ["rs429358_T", "rs7412_C"]) 

# pull cases
alz_eid, _, _ = dem.pull_dementia_cases(ddf, alzheimers_only = True) # a tuple of both_eid, date_df, exclude_df

dem_eid , _ , _ = dem.pull_dementia_cases(ddf, alzheimers_only = False) # a tuple of both_eid, date_df, exclude_df

# add an extra column to the df with whether the patient is a case (1) or control (0) or neither (2)
df['groups'] = 0 

# assign 2 to excluded patients
df.loc[df['IID'].isin(dem_eid), 'groups'] = 2.0

# assign 1 to alz patients 
df.loc[df['IID'].isin(alz_eid), 'groups'] = 1.0

# drop excluded
df = df[df.groups.isin([0, 1])]

# BEGIN TRAINING
#args
args = parse_args()
fold_index = args.fold_index
# experiment = args.experiment

# defining x, y
X = df.drop(columns = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE', 'groups'])
y = df.values[:, -1]

results_dir = f"./results/polygenic_risk_score/{fold_index}"
# results_dir = f"./results/{fold_index}_65up/"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# logging
logging.basicConfig(filename=f"{results_dir}/logging.txt", level=logging.INFO)
logging.info(
        "Running experiment %s with IDP index %d and fold index %d",
    )
logging.info("X shape: %s", X.shape)
logging.info("y shape: %s", y.shape)

# settings func
def settings_automl(time_budget, metric, model):
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
            "metric": 'log_loss',
            "n_jobs": -1,
            "eval_method": "cv",
            "n_splits": 5,
            "early_stop": True,
            "log_training_metric": True,
            "model_history": True,
            "seed": 1234321,
            "estimator_list": [model],
    }

    return automl_settings

# skf and ml
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234321)

logging.info(f"Starting the k-fold cross-validation")
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    if fold != fold_index:
        continue
    print('folds')
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print('training')
    automl = AutoML()
    automl_settings = settings_automl(FLAML_TIME_BUDGET, metric="log_loss", model="lgbm")
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
        f"{results_dir}/test_labels_predictions.parquet", index=False
    )

    # save the train set predictions
    y_pred = automl.predict_proba(X_train)
    results = pd.DataFrame({"y_train": y_train, "y_pred": y_pred[:,1]})
    results.to_parquet(
        f"{results_dir}/train_labels_predictions.parquet", index=False
    )