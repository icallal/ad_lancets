"""AutoML experiments for proteomics IDP analysis."""

import argparse
import logging
import os
import sys
import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime

sys.path.append("../../../rfb/code/ukb_func")
import ml_utils
from ukb_utils import get_protein_lookup

sys.path.append("../../../rfb/code/ukbiobank")
import ml_experiments as ml_exp
import utils
from calculate_performance import calculate_metrics

# from df_utils import pull_columns_by_prefix, pull_columns_by_suffix, pull_rows_with_values, row_contains_values

import matplotlib.pyplot as plt
from flaml import AutoML
from flaml.automl.data import get_output_from_log

from sklearn.metrics import r2_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idp_index", type=int, required=True)
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
    )
    return parser.parse_args()


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
            "task": "regression",
            "train_full": True,
            "n_jobs": -1,
            "train_best": True,
        }

    else:
        automl_settings = {
            "task": "regression",
            "time_budget": time_budget,
            "metric": metric,
            "n_jobs": -1,
            "eval_method": "cv", # cross validation
            "n_splits": 5,
            "early_stop": True,
            "log_training_metric": True,
            "model_history": True,
            "seed": 1234321,
            "estimator_list": ["lgbm"],
        }

    return automl_settings


def iterative_fs_inference(
    automl,
    automl_settings,
    config_history,
    X_train,
    y_train,
    X_test,
    y_test,
    results_dir,
):
    """
    Perform iterative feature selection and inference using AutoML.

    Args:
        automl_settings (dict): Settings for AutoML.
        config_history (list): List of configuration history.
        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data labels.
        X_test (pandas.DataFrame): Test data features.
        y_test (pandas.Series): Test data labels.
        results_dir (str): Path to the directory.

    Returns:
        tuple: A tuple containing the following:
            - train_labels_l (list): List of training data labels for each iteration.
            - train_probas_l (list): List of training data probabilities for each iteration.
            - test_labels_l (list): List of test data labels for each iteration.
            - test_probas_l (list): List of test data probabilities for each iteration.
            - train_res_l (list): List of training data results for each iteration.
            - test_res_l (list): List of test data results for each iteration.
    """
    # train_labels_l = []
    # train_preds_l = []

    # test_labels_l = []
    # test_preds_l = []

    # train_res_l = []
    # test_res_l = []

    top_feature_names, fi = ml_exp.get_top_features(automl)
    # get protein names from the top feature names
    protein_lookup = get_protein_lookup("../../../rfb/metadata/coding143.tsv")
    tf_coding = [int(x.split("-")[0]) for x in top_feature_names]
    subset = protein_lookup[protein_lookup["coding"].isin(tf_coding)]
    subset = subset.set_index("coding")
    subset = subset.loc[tf_coding]
    subset["feature_importance"] = fi
    subset.columns = ["symbol", "protein_name", "feature_importance"]
    subset = subset[subset["feature_importance"] > 0]

    # save the subset to a csv file
    subset.to_csv(f"{results_dir}/top_feature_names.csv", index=False)

    tflist = []

    if not os.path.exists(f"{results_dir}/individual_results/"):
        os.makedirs(f"{results_dir}/individual_results/")

    # Initialize lists to store results
    train_res_list = []
    test_res_list = []

    num_feats = min(500, subset.shape[0])
    for j, tf in enumerate(top_feature_names[:num_feats]):
        tflist.append(tf)
        current_time = datetime.now().time()
        print(f"Running top {j+1} features: {tflist}, {current_time}")

        X_train_sub = X_train.loc[:, tflist]
        X_test_sub = X_test.loc[:, tflist]

        automl = AutoML()
        automl.retrain_from_log(X_train=X_train_sub, y_train=y_train, **automl_settings)

        current_time = datetime.now().time()
        print(f"Done fitting model for top {j+1} variables. {current_time}")

        # series_automl = pd.Series(
        #     [
        #         config_history[-1]["Best Learner"],
        #         config_history[-1]["Best Hyper-parameters"],
        #         tflist,
        #     ],
        #     index=["model", "hyperparams", "features"],
        # )

        train_preds = automl.predict(X_train_sub)
        # train_res, threshold = ml_utils.calc_results(y_train, train_preds, beta=1)
        # train_res = pd.concat([series_automl, train_res])
        # train_res_l.append(train_res)

        # if j == 0:
        #     train_labels_l.append(y_train)
        # train_preds_l.append(train_preds)

        test_preds = automl.predict(X_test_sub)
        # test_res = ml_utils.calc_results(
        #     y_test, test_preds, beta=1, threshold=threshold
        # )
        # test_res = pd.concat([series_automl, test_res])
        # test_res_l.append(test_res)

        # if j == 0:
        #     test_labels_l.append(y_test)
        # test_preds_l.append(test_preds)

        # save the test set predictions
        results = pd.DataFrame({"y_test": y_test, "y_pred": test_preds})
        results.to_parquet(
            f"{results_dir}/individual_results/test_labels_predictions_top_{j+1}.parquet",
            index=False,
        )
        test_metrics = calculate_metrics(results["y_test"], results["y_pred"])
        test_res_list.append(test_metrics)

        # save the train set predictions
        results = pd.DataFrame({"y_train": y_train, "y_pred": train_preds})
        results.to_parquet(
            f"{results_dir}/individual_results/train_labels_predictions_top_{j+1}.parquet",
            index=False,
        )
        train_metrics = calculate_metrics(results["y_train"], results["y_pred"])
        train_res_list.append(train_metrics)

    # convert train_res_list and test_res_list to dataframes
    train_res_df = pd.DataFrame(train_res_list)
    test_res_df = pd.DataFrame(test_res_list)

    # save the dataframes to parquet files
    train_res_df.to_csv(f"{results_dir}/train_results.csv")
    test_res_df.to_csv(f"{results_dir}/test_results.csv")

    # return (
    #     train_labels_l,
    #     train_preds_l,
    #     test_labels_l,
    #     test_preds_l,
    #     train_res_l,
    #     test_res_l,
    # )


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

    ml_utils.save_labels_probas(
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


def find_top_proteins_omicspred(idp_index, fold_index, experiment):
    """
    Find the top proteins for a given IDP index and fold index.
    """

    if experiment == "proteins_omicspred":
        original_experiment = "proteins_alone"
        fs_experiment = "fs_proteins_alone"
    else:
        print("Experiment not found")
        exit()

    full_res = pd.read_parquet(
        f"../../results/{original_experiment}/individual_results/idp_{idp_index}/fold_{fold_index}/test_labels_predictions.parquet"
    )
    r2 = r2_score(full_res.y_test, full_res.y_pred)
    print(f"Fold {fold_index} R2: {r2}")

    fi_res = pd.read_csv(
        f"../../results/{fs_experiment}/individual_results/idp_{idp_index}/fold_{fold_index}/test_results.csv"
    )
    prot_list = pd.read_csv(
        f"../../results/{fs_experiment}/individual_results/idp_{idp_index}/fold_{fold_index}/top_feature_names.csv"
    )

    pct_full = [x / r2 for x in fi_res.r2_score]
    # find the first value in pct_full that is over 0.9
    for i, x in enumerate(pct_full):
        if x > 0.98:
            print(f"{i} proteins, R2 score: {x * r2}")
            top_proteins = prot_list.iloc[: i + 1].symbol.tolist()
            break

    protein_lookup = get_protein_lookup("../../../rfb/metadata/coding143.tsv")
    subset = protein_lookup[protein_lookup.part_1.isin(top_proteins)]
    protein_columns = [f"{x}-0" for x in subset.coding]

    return top_proteins, protein_columns


def extract_protGS(prot_id):
    omicpredIDs = pd.read_csv(
        "/n/groups/patel/shakson_ukb/UK_Biobank/Data/OMICSPRED/UKB_Olink_multi_ancestry_models_val_results_portal.csv",
        sep="\t",
    )
    dir_protgs = "/n/groups/patel/IGLOO/UKB/ProtGS/"
    
    # check if protID is in the omicpredIDs dataframe GENE column
    if prot_id not in omicpredIDs.Gene.values:
        print(f"Protein {prot_id} not found in omicpredIDs dataframe")
        return None
    else:
        op_id = omicpredIDs[omicpredIDs.Gene == prot_id].OMICSPRED_ID.values[0]

    op_file = f"{op_id}.sscore"
    prot_gs = pd.read_csv(f"{dir_protgs}{op_file}", sep="\t")
    prot_gs.columns = ["eid", f"{prot_id}_GS"]

    # Convert dash to underscore:
    prot_gs.columns = prot_gs.columns.str.replace("-", "_")
    return prot_gs


def main():
    """Main entry point for running the AutoML experiment."""
    args = parse_args()
    idp_index = args.idp_index
    fold_index = args.fold_index
    experiment = args.experiment
    
    FLAML_TIME_BUDGET = 3600

    print(os.getcwd())

    print(f"Running experiment with IDP index {idp_index} and fold index {fold_index}")
    prot = pd.read_parquet("../../tidy_data/proteomics/proteomics_clean.parquet")
    idp = pd.read_parquet("../../tidy_data/idp/idp_instance2_clean.parquet")

    # create a directory for the results if it doesn't exist
    # replace spaces and special characters with underscores
    results_dir = f"../../results/{experiment}/individual_results/idp_{idp_index}/fold_{fold_index}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if "fs" in experiment:
        original_experiment = experiment[3:]  # remove the fs_ prefix
        print(
            f"Running iterative feature selection experiment on {original_experiment}"
        )

        original_results_dir = f"../../results/{original_experiment}/individual_results/idp_{idp_index}/fold_{fold_index}"
    else:
        print(f"Running standard experiment: {experiment}")
        # save a text file with the idp name if it doesn't exist
        if not os.path.exists(f"{results_dir}/../idp_name.txt"):
            with open(f"{results_dir}/../idp_name.txt", "w") as f:
                f.write(idp.columns[idp_index])

    # subset the eids and the idp column
    idp = idp.iloc[:, [0, idp_index]]
    idp = idp.dropna()

    prot = prot[prot["eid"].isin(idp["eid"])]
    print(prot.shape, idp.shape)
    if "asp" in experiment:
        logging.info("Adding age and sex information to the data")
        # load age and sex information
        age_sex = pd.read_parquet(
            "../../../rfb/tidy_data/UKBiobank/dementia/proteomics/X.parquet"
        )
        age_sex = age_sex.loc[:, ["eid", "21003-0.0", "31-0.0_1.0"]]
        prot = prot.merge(age_sex, on="eid", how="left")
        logging.info("Age and sex information added to the data")
        logging.info(prot.columns)

    if "omicspred" in experiment:
        logging.info("Adding omicspred scores to the data")
        top_proteins, protein_columns = find_top_proteins_omicspred(
            idp_index, fold_index, experiment
        )

        prot = prot.loc[:, ["eid"] + protein_columns]
        for prot_id in top_proteins:
            prot_gs = extract_protGS(prot_id)
            if prot_gs is None:
                print(f"Protein {prot_id} not found in omicpredIDs dataframe")
                continue
            prot = prot.merge(prot_gs, on="eid", how="left")
        logging.info(f"Omicspred scores added to the data: {top_proteins}")
        logging.info(prot.columns)

    prot = prot.set_index("eid").loc[idp.set_index("eid").index]
    print(prot.shape, idp.shape)

    # confirm that the order of the eids are the same
    if np.all(prot.index == idp.eid.values):
        print("The order of the eids are the same")
    else:
        print("The order of the eids are not the same")
        # stop code here
        exit()

    X = prot
    y = idp.iloc[:, 1]

    # set up logging
    logging.basicConfig(filename=f"{results_dir}/logging.txt", level=logging.INFO)
    logging.info(
        "Running experiment %s with IDP index %d and fold index %d",
        experiment,
        idp_index,
        fold_index,
    )
    logging.info("X shape: %s", X.shape)
    logging.info("y shape: %s", y.shape)

    try:
        # Initialize the k-fold cross-validator
        kf = KFold(n_splits=5, shuffle=True, random_state=1234321)

        logging.info(f"Starting the k-fold cross-validation")
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            if fold != fold_index:
                continue
            logging.info(f"Starting fold {fold}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            logging.info(f"Training the model")

            automl = AutoML()
            automl_settings = settings_automl(
                experiment, FLAML_TIME_BUDGET, metric="mse", model="lgbm"
            )
            print(automl_settings)

            if "fs" in experiment:
                automl_settings["log_file_name"] = (
                    f"{original_results_dir}/results_log.json"
                )

                print(
                    f"Retraining best model to do feature selection: {datetime.now().time()}"
                )
                automl.retrain_from_log(
                    X_train=X_train, y_train=y_train, **automl_settings
                )
                print(f"Done retraining model: {datetime.now().time()}")

                (
                    time_history,
                    best_valid_loss_history,
                    valid_loss_history,
                    config_history,
                    metric_history,
                ) = get_output_from_log(
                    filename=f"{original_results_dir}/results_log.json",
                    time_budget=FLAML_TIME_BUDGET,
                )

                # (
                #     train_labels_l,
                #     train_probas_l,
                #     test_labels_l,
                #     test_probas_l,
                #     train_res_l,
                #     test_res_l,
                # ) =
                iterative_fs_inference(
                    automl,
                    automl_settings,
                    config_history,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    results_dir,
                )

                # save_fs_results(
                #     results_dir,
                #     train_labels_l,
                #     train_probas_l,
                #     test_labels_l,
                #     test_probas_l,
                #     train_res_l,
                #     test_res_l,
                # )

            else:
                automl_settings["log_file_name"] = f"{results_dir}/results_log.json"
                print(f"Fitting model: {datetime.now().time()}")
                automl.fit(X_train, y_train, **automl_settings)
                print(f"Done fitting model: {datetime.now().time()}")

                logging.info(f"Saving the model: {datetime.now().time()}")
                # save the model
                best_model = automl.model.estimator

                # Save just the best model
                joblib.dump(best_model, f"{results_dir}/flaml_best_model.joblib")

                logging.info(f"Saving the predictions: {datetime.now().time()}")
                # save the test set predictions
                y_pred = automl.predict(X_test)
                results = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
                results.to_parquet(
                    f"{results_dir}/test_labels_predictions.parquet", index=False
                )

                # save the train set predictions
                y_pred = automl.predict(X_train)
                results = pd.DataFrame({"y_train": y_train, "y_pred": y_pred})
                results.to_parquet(
                    f"{results_dir}/train_labels_predictions.parquet", index=False
                )

            logging.info(f"Finished fold {fold}: {datetime.now().time()}")

        logging.info(f"Finished the k-fold cross-validation: {datetime.now().time()}")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)  # Includes traceback

    finally:
        logging.info(f"Finished the experiment: {datetime.now().time()}")


if __name__ == "__main__":
    main()
