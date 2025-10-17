import pandas as pd
import numpy as np
import dementia_utils as dem

import argparse
import json
import logging
import os
import sys
import joblib
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from mlutils import calc_results
from mlutils import pick_threshold
from mlutils import save_labels_probas

from flaml import AutoML

FIRST_TIME_BUDGET = 3600
SECOND_TIME_BUDGET = 4800

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument( 
        "--experiment",
        type=str,
        help="""Experiment name. Options:
        'fs_genetics', 
        'fs_gentics_demographics',
        'fs_genetics_demographics_lancet'""",
        required = False
    )

    args = parser.parse_args()

    fold_index = args.fold_index

    # Parse the arguments
    experiment = args.experiment
    
    return (
        fold_index,
        experiment,
    )

def get_top_features(automl):
    """
    Extracts the top features from an AutoML model.
    Parameters:
    automl (object): The AutoML model object.
    Returns:
    list: A list of the top features from the AutoML model.
    """
    if len(automl.feature_importances_) == 1:
        feature_names = np.array(automl.feature_names_in_)[
            np.argsort(abs(automl.feature_importances_[0]))[::-1]
        ]
        fi = automl.feature_importances_[0][
            np.argsort(abs(automl.feature_importances_[0]))[::-1]
        ]
    else:
        feature_names = np.array(automl.feature_names_in_)[
            np.argsort(abs(automl.feature_importances_))[::-1]
        ]
        fi = automl.feature_importances_[
            np.argsort(abs(automl.feature_importances_))[::-1]
        ]

    return feature_names, fi


# alzheimers cases only? or not...

def pull_alzheimer_only(df, ddf):
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

    return df

def encode_apoe(df):
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

    return df

# the dfs to read in? 
def load_datasets(): 
    # df for which snps
    df = pd.read_parquet('snp_parquets/raw_unlinkedsnps_AD.parquet', engine = 'fastparquet') # LDE snps
    df_apoe = pd.read_parquet('snp_parquets/raw_apoe_AD.parquet', engine = 'fastparquet') # apoe only
    df = df.merge(df_apoe, on="IID", how="left") # add apoe snps to LDE snps by merging on IID
    
    # which ages
    ddf = pd.read_parquet('allcausedementia.parquet', engine = 'fastparquet') # age, sex, all-patients
 
    # assign groups 
    df = pull_alzheimer_only(df, ddf)

    drop_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE', 'groups']
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df['groups'].values
    
    return X, y

def get_dir_path(experiment):  
    root = "./results_fs"

    path = f"{root}/{experiment}"  

    return path

lancet_vars = [
        "4700-0.0",
        "5901-0.0",
        "30780-0.0",
        "head_injury",
        "22038-0.0",
        "20161-0.0",
        "alcohol_consumption",
        "hypertension",
        "obesity",
        "diabetes",
        "hearing_loss",
        "depression",
        "freq_friends_family_visit",
        "24012-0.0",
        "24018-0.0",
        "24019-0.0",
        "24006-0.0",
        "24015-0.0",
        "24011-0.0",
        "2020-0.0_-3.0",
        "2020-0.0_-1.0",
        "2020-0.0_0.0",
        "2020-0.0_1.0",
        "2020-0.0_nan",
    ]



def subset_experiment_vars(X, experiment):
    lancets = pd.read_parquet('../../../randy/rfb/tidy_data/UKBiobank/dementia/lancet2024/lancet2024_preprocessed.parquet', engine = 'fastparquet')
    demographics = pd.read_parquet('demographics.parquet', engine='fastparquet')

    # experiment variables
    experiment_vars = {

        "fs_genetics":[], 
    
        "fs_genetics_demographics": 
            [
                "curr_age",
                "31-0.0",
                #"apoe",
                '6138-0.0', # 'max_educ_complete'
                "845-0.0",
                "21000-0.0",
            ],
    
        "fs_genetics_demographics_lancet":
            [
                "curr_age",
                "31-0.0",
                #"apoe",
                '6138-0.0',
                "845-0.0",
                "21000-0.0",
            ]   
    }

    features_to_add = experiment_vars[experiment]
    # Subset ddf to only include patients present in X (assuming 'eid' is the patient identifier)
    demo_sub = (
        demographics.loc[demographics['eid'].isin(X.index), 
                          features_to_add]
    )

    X = X.merge(demo_sub, 
                left_index=True, 
                right_index=True, 
                how='left')

    if experiment == "fs_genetics_demographics" or experiment == "fs_genetics_demographics_lancet":
        lancet_cols = lancets.columns.intersection(lancet_vars)
        lanc_sub = (
            lancets.loc[lancets['eid'].isin(X.index), 
                        lancet_cols]
        )
    
        X = X.merge(lanc_sub,
                        left_index=True,
                        right_index=True,
                        how='left')
  
    return X

def subset_train_test(X, y, results_dir, fold_index):
    # logging
    logging.basicConfig(filename=f"{results_dir}/logging.txt", level=logging.INFO)
    logging.info(
            "Running fold index {fold index}",
        )
    logging.info("X shape: %s", X.shape)
    logging.info("y shape: %s", y.shape)

    # creating the folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234321)

    logging.info(f"Starting the k-fold cross-validation")
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        if fold != fold_index:
            continue
        print('folds')
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test

def settings_automl(experiment, time_budget, metric, model):
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
    if "fs_" in experiment:
        automl_settings = {
            "task": "classification",
            "train_full": True,
            "n_jobs": -1,
            "train_best": True,
        }

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
            "estimator_list": [model],
    }

    return automl_settings


def get_top_features(automl):
    """
    Extracts the top features from an AutoML model.
    Parameters:
    automl (object): The AutoML model object.
    Returns:
    list: A list of the top features from the AutoML model.
    """
    if len(automl.feature_importances_) == 1:
        feature_names = np.array(automl.feature_names_in_)[
            np.argsort(abs(automl.feature_importances_[0]))[::-1]
        ]
        fi = automl.feature_importances_[0][
            np.argsort(abs(automl.feature_importances_[0]))[::-1]
        ]
    else:
        feature_names = np.array(automl.feature_names_in_)[
            np.argsort(abs(automl.feature_importances_))[::-1]
        ]
        fi = automl.feature_importances_[
            np.argsort(abs(automl.feature_importances_))[::-1]
        ]

    return feature_names, fi

def save_fs_results(
    directory_path,
    train_labels_l,
    train_probas_l,
    test_labels_l,
    test_probas_l,
    train_res_l,
    test_res_l,
):
    """
    Save the results of a machine learning experiment to files.

    Parameters:
    - directory_path (str): The path to the directory where the files will be saved.
    - train_labels_l (list): A list of training labels.
    - train_probas_l (list): A list of training probabilities.
    - test_labels_l (list): A list of test labels.
    - test_probas_l (list): A list of test probabilities.
    - train_res_l (list): A list of training results.
    - test_res_l (list): A list of test results.

    Returns:
    None
    """

    save_labels_probas(
        directory_path,
        train_labels_l,
        train_probas_l,
        test_labels_l,
        test_probas_l,
    )

    train_df = pd.concat(train_res_l, axis=1).T
    train_df.to_csv(f"{directory_path}/training_results.csv")

    test_df = pd.concat(test_res_l, axis=1).T
    test_df.to_csv(f"{directory_path}/test_results.csv")


def save_feature_importance(automl, directory_path):
    """
    Save the feature importance from an AutoML model to a parquet file.
    Parameters:
    automl (object): The AutoML model object that contains feature importances and feature names.
    directory_path (str): The directory path where the parquet file will be saved.
    region_index (int): The index of the region for which the feature importance is being saved.
    Returns:
    None
    The function extracts feature importances and their corresponding feature names from the AutoML model,
    sorts them in descending order of importance, and saves the result as a parquet file in the specified directory.
    """
    feature_names, fi = get_top_features(automl)

    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
    fi_df.to_parquet(
        f"{directory_path}/feature_importance.parquet"
    )

def retrain_on_top_k_features(automl_log_path, X_train, y_train, k=100):
    automl_settings = {
        "time_budget": 0,  # dummy
        "log_file_name": automl_log_path,
        "task": "classification",
        "metric": "log_loss",
    }

    # Load and retrain model
    automl = AutoML()
    automl.retrain_from_log(X_train=X_train, y_train=y_train, **automl_settings)

    top_k_features, _ = get_top_features(automl)
    top_k_features = top_k_features[:k]
    print(f"Top {k} features: {top_k_features}")

    # Final retrain on just top-k features
    X_train_sub = X_train[top_k_features]
    final_model = AutoML()
    final_model.retrain_from_log(X_train=X_train_sub, y_train=y_train, **automl_settings)

    return final_model, top_k_features


# def iterative_fs_inference(
#     automl,
#     automl_settings,
#     config_history,
#     X_train,
#     y_train,
#     X_test,
#     y_test
# ):
#     """
#     Perform iterative feature selection and inference using AutoML.

#     Args:
#         automl_settings (dict): Settings for AutoML.
#         config_history (list): List of configuration history.
#         X_train (pandas.DataFrame): Training data features.
#         y_train (pandas.Series): Training data labels.
#         X_test (pandas.DataFrame): Test data features.
#         y_test (pandas.Series): Test data labels.
#         directory_path (str): Path to the directory.
#         region_index (int): Index of the region.
#         region (str): Name of the region.

#     Returns:
#         tuple: A tuple containing the following:
#             - train_labels_l (list): List of training data labels for each iteration.
#             - train_probas_l (list): List of training data probabilities for each iteration.
#             - test_labels_l (list): List of test data labels for each iteration.
#             - test_probas_l (list): List of test data probabilities for each iteration.
#             - train_res_l (list): List of training data results for each iteration.
#             - test_res_l (list): List of test data results for each iteration.
#     """
#     train_labels_l = []
#     train_probas_l = []

#     test_labels_l = []
#     test_probas_l = []

#     train_res_l = []
#     test_res_l = []

#     top_feature_names, _ = get_top_features(automl)

#     tflist = []
#     for j, tf in enumerate(top_feature_names[:100]):
#         tflist.append(tf)
#         current_time = datetime.now().time()
#         print(f"Running top {j+1} features: {tflist}, {current_time}")

#         X_train_sub = X_train.loc[:, tflist]
#         X_test_sub = X_test.loc[:, tflist]

#         automl = AutoML()
#         automl.retrain_from_log(X_train=X_train_sub, y_train=y_train, **automl_settings)

#         current_time = datetime.now().time()
#         print(
#             f"Done fitting model: {current_time}"
#         )

#         series_automl = pd.Series(
#             [
#                 config_history[-1]["Best Hyper-parameters"],
#                 tflist,
#             ],
#             index=["hyperparams", "features"],
#         )

#         train_probas = automl.predict_proba(X_train_sub)[:, 1]
#         train_res, threshold = calc_results(y_train, train_probas, beta=1)
#         train_res = pd.concat([series_automl, train_res])
#         train_res_l.append(train_res)

#         if j == 0:
#             train_labels_l.append(y_train)
#         train_probas_l.append(train_probas)

#         test_probas = automl.predict_proba(X_test_sub)[:, 1]
#         test_res = calc_results(
#             y_test, test_probas, beta=1, threshold=threshold
#         )
#         test_res = pd.concat([series_automl, test_res])
#         test_res_l.append(test_res)

#         if j == 0:
#             test_labels_l.append(y_test)
#         test_probas_l.append(test_probas)

#     return (
#         train_labels_l,
#         train_probas_l,
#         test_labels_l,
#         test_probas_l,
#         train_res_l,
#         test_res_l,
#     )

def save_fs_results(
    directory_path,
    train_labels_l,
    train_probas_l,
    test_labels_l,
    test_probas_l,
    train_res_l,
    test_res_l,
):
    """
    Save the results of a machine learning experiment to files.

    Parameters:
    - directory_path (str): The path to the directory where the files will be saved.
    - train_labels_l (list): A list of training labels.
    - train_probas_l (list): A list of training probabilities.
    - test_labels_l (list): A list of test labels.
    - test_probas_l (list): A list of test probabilities.
    - train_res_l (list): A list of training results.
    - test_res_l (list): A list of test results.
    - region_index (int): The index of the region.

    Returns:
    None
    """

    save_labels_probas(
        directory_path,
        train_labels_l,
        train_probas_l,
        test_labels_l,
        test_probas_l
    )

    train_df = pd.concat(train_res_l, axis=1).T
    train_df.to_csv(f"{directory_path}/training_results.csv")

    test_df = pd.concat(test_res_l, axis=1).T
    test_df.to_csv(f"{directory_path}/test_results.csv")

def save_results(
    directory_path, automl, X_train, y_train, X_test, y_test, region, region_index
):
    """
    Save the results of an AutoML experiment to CSV files.
    Parameters:
    directory_path (str): The directory where the results will be saved.
    automl (object): The AutoML object that contains the trained model and its configurations.
    X_train (pd.DataFrame): The training feature set.
    y_train (pd.Series): The training labels.
    X_test (pd.DataFrame): The test feature set.
    y_test (pd.Series): The test labels.
    region (str): The region name or identifier.
    region_index (int): The index of the region.
    Returns:
    None
    This function performs the following steps:
    1. Initializes lists to store training and test labels and probabilities.
    2. Creates a pandas Series with the AutoML model and its best configuration.
    3. Predicts probabilities for the training set and calculates results.
    4. Appends the training labels and probabilities to their respective lists.
    5. Predicts probabilities for the test set and calculates results using the same threshold as the training set.
    6. Appends the test labels and probabilities to their respective lists.
    7. Saves the labels and probabilities to files.
    8. Saves the training results to a CSV file, appending to the file if it already exists.
    9. Saves the test results to a CSV file, appending to the file if it already exists.
    """
    # set up lists to store results
    train_labels_l = []
    train_probas_l = []

    test_labels_l = []
    test_probas_l = []

    series_automl = pd.Series(
        [region_index, region, automl.best_estimator, automl.best_config],
        index=["region_index", "region", "model", "hyperparams"],
    )

    train_probas = automl.predict_proba(X_train)[:, 1]

    train_res, threshold = calc_results(y_train, train_probas, beta=1)
    train_res = pd.concat([series_automl, train_res])
    train_labels_l.append(y_train)
    train_probas_l.append(train_probas)

    test_probas = automl.predict_proba(X_test)[:, 1]
    test_res = calc_results(y_test, test_probas, beta=1, threshold=threshold)

    test_res = pd.concat([series_automl, test_res])
    test_labels_l.append(y_test)
    test_probas_l.append(test_probas)

    save_labels_probas(
        directory_path,
        train_labels_l,
        train_probas_l,
        test_labels_l,
        test_probas_l,
        other_file_info=f"_region_{region_index}",
    )

    train_df = pd.DataFrame(train_res).T
    # Specify the path to your CSV file
    file_path = f"{directory_path}/training_results.csv"

    if os.path.exists(file_path):
        # If file exists, read it into a DataFrame and append the new data
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, train_df])
        combined_df.to_csv(file_path, index=False)
    else:
        # If file does not exist, simply write the new data to a CSV
        train_df.to_csv(file_path, index=False)

    test_df = pd.DataFrame(test_res).T
    # Specify the path to your CSV file
    file_path = f"{directory_path}/test_results.csv"

    if os.path.exists(file_path):
        # If file exists, read it into a DataFrame and append the new data
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, test_df])
        combined_df.to_csv(file_path, index=False)
    else:
        # If file does not exist, simply write the new data to a CSV
        test_df.to_csv(file_path, index=False)

def get_output_from_log(filename, time_budget=None):
    """
    Reads the FLAML log file and returns the config history.
    """
    with open(filename, "r") as f:
        log_data = json.load(f)
    # FLAML's log is a list of dicts, each with a config and results
    # You may want to return the whole log, or just the best config
    return log_data
    
def main(): 
    # parse the arguments
    
    (   
        fold_index,
        experiment
    ) = parse_args()
    
    X, y = load_datasets()

    # create results dir
    results_dir = get_dir_path(experiment) + automl.retrain_from_log(X_train=X_train, y_train=y_train, **automl_settings)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # choosing experimental variables
    print('subsetting experiment variables')
    X = subset_experiment_vars(X, experiment)
    print(f"X shape after subsetting: {X.shape}")

    # split + train
    print('splitting')
    X_train, y_train, X_test, y_test = subset_train_test(X, y, results_dir, fold_index)

    print('training')
    automl = AutoML()
    automl_settings = settings_automl(experiment, FIRST_TIME_BUDGET, metric="log_loss", model='lgbm')
    print(automl_settings)

    print('feature selection')
    if "fs_" in experiment:
        automl_settings["log_file_name"] = (
            f"{results_dir}/results_log.json"
        )
        print(f"Running AutoML to create log: {datetime.now().time()}")
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
        print(f"Done initial AutoML fit: {datetime.now().time()}")

        print(f"Retraining best model to do feature selection: {datetime.now().time()}")
        automl.retrain_from_log(X_train=X_train, y_train=y_train, **automl_settings)
        print(f"Done retraining model: {datetime.now().time()}")

        log_path = f"{results_dir}/results_log.json"
        final_model, top_k_features = retrain_on_top_k_features(log_path, X_train, y_train, k=100)

        # Save final model
        joblib.dump(final_model.model, f"{results_dir}/final_model_top100.joblib")

        # Evaluate on test
        X_test_sub = X_test[top_k_features]
        test_probas = final_model.predict_proba(X_test_sub)[:, 1]
        test_res = calc_results(y_test, test_probas, beta=1)

        # Save test results
        test_df = pd.DataFrame({
            "true": y_test,
            "proba": test_probas
        })
        test_df.to_parquet(f"{results_dir}/test_preds_top100.parquet", index=False)

        print("Final model and test predictions saved.")

 
if __name__ == "__main__":
    main()
    # parse_args()
    # main() # this is the main function that runs the experiment