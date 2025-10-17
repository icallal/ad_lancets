import pandas as pd
import numpy as np
import dementia_utils as dem

import argparse
import logging
import os
import sys
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from mlutils import calc_results
from mlutils import pick_threshold

from flaml import AutoML

FLAML_TIME_BUDGET = 3600

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument( # this modifies the field ids
        "--experiment",
        type=str,
        help="""Experiment name. Options:
        'none' - no demographic information
        'age_alone' - only age
        'all_demographics' - age, sex, apoe, education?
        'age_sex_lancet2024' - lancet factors + age and sex
        'demographics_lancet2024' - lancet factors + all demographics
        """,
        required = False
    )

    # these modify the patient-snp arrays
    parser.add_argument("--age_cutoff", type=int, default=None, help="Options: False (allages), True (65up)", required=True)
    parser.add_argument("--snps", type=str, default=None, help="Options: 'all_snps', 'apoe', 'only_snps', 'LDE', 'polygenic_risk_score', 'none', 'apoe_stratified', 'preprint_only'", required=True)
    parser.add_argument("--alzheimers_only", type=int, default=None, help="Options: True ('AD'), False('ACD')", required=True)
    parser.add_argument("--model", type=str, default=None, help="Model to use. Options: 'lgbm', 'lrl1'", required=True)

    args = parser.parse_args()

    fold_index = args.fold_index

    # Parse the arguments
    experiment = args.experiment

    if args.age_cutoff == 0:
        age_cutoff = False # allages
    elif args.age_cutoff == 1: 
        age_cutoff = True # 65up

    snps = args.snps
    
    if args.alzheimers_only == 0:
        alzheimers_only = False
    elif args.alzheimers_only == 1:
        alzheimers_only = True
    else:
        print("predict_alzheimers_only must be 0 or 1")
        sys.exit()
    
    model = args.model
    
    return (
        fold_index,
        experiment,
        age_cutoff,
        snps, 
        alzheimers_only,
        model
    )

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


def pull_all_dementia(df, ddf):
    # all dementias
    dem_eid , _ , _ = dem.pull_dementia_cases(ddf, alzheimers_only = False) # a tuple of both_eid, date_df, exclude_df

    # add an extra column to the df with whether the patient is a case (1) or control (0) or neither (2)
    df['groups'] = 0 

    # assign 2 to excluded patients
    df.loc[df['IID'].isin(dem_eid), 'groups'] = 1.0

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

def encode_education(df, col='6138-0.0', prefix='education'):
    """
    One-hot encode the specified education column in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        col (str): Column name to encode.
        prefix (str): Prefix for new columns.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded education columns.
    """
    # One-hot encode, treating NaN as a separate category if needed
    dummies = pd.get_dummies(df[col], prefix=prefix, dummy_na=False)
    dummies = dummies.astype(int)
    df = pd.concat([df, dummies], axis=1)
    return df


# the dfs to read in? 
def load_datasets(snps, age_cutoff=True, alzheimers_only=True): 
    # df for which snps
    
    if snps == 'apoe':
        df = pd.read_parquet('./snp_parquets/raw_apoe_AD.parquet', engine = 'fastparquet') # apoe only
    elif snps == 'all_snps':
        df = pd.read_parquet('./snp_parquets/raw_allsnps_AD.parquet', engine = 'fastparquet') # all snps
        # encode apoe
        df = encode_apoe(df)
    elif snps == 'LDE':
        df = pd.read_parquet('./snp_parquets/raw_unlinkedsnps_AD.parquet', engine = 'fastparquet') # LDE snps
        df_apoe = pd.read_parquet('./snp_parquets/raw_apoe_AD.parquet', engine = 'fastparquet') # apoe only
        df = df.merge(df_apoe, on="IID", how="left") # add apoe snps to LDE snps by merging on IID
    elif snps == 'only_snps':
        df = pd.read_parquet('./snp_parquets/raw_unlinkedsnps_AD.parquet', engine = 'fastparquet') # LDE snps
        if 'rs429358_T' in df.columns or 'rs7412_C' in df.columns:
            df = df.drop(columns=[col for col in ['rs429358_T', 'rs7412_C'] if col in df.columns])
    elif snps == 'none':
        # just getting the id's, but i'm sure theres a better way to do this
        df = pd.read_parquet('./snp_parquets/raw_apoe_AD.parquet', engine = 'fastparquet') 
        df = df['IID'].to_frame() # just the id's, no snps
    elif snps == 'apoe_stratified': 
        df = pd.read_parquet('./snp_parquets/raw_apoestrat_AD.parquet', engine = 'fastparquet')
        df_apoe = pd.read_parquet('./snp_parquets/raw_apoe_AD.parquet', engine = 'fastparquet') # apoe only
        df = df.merge(df_apoe, on="IID", how="left") # add apoe snps to LDE snps by merging on IID
    elif snps == 'polygenic_risk_score':
        df = pd.read_parquet('../../../randy/proj_idp/tidy_data/prs_Alz/prs_Alz.parquet', engine = 'fastparquet')
        if 'eid' in df.columns:
            df = df.rename(columns={'eid': 'IID'}) # polygenic risk score
    elif snps == 'preprint_only':
        df = pd.read_parquet('./snp_parquets/raw_preprint_snps.parquet', engine = 'fastparquet')
        df_apoe = pd.read_parquet('./snp_parquets/raw_apoe_AD.parquet', engine = 'fastparquet')
        df = df.merge(df_apoe, on="IID", how="left")
    else: 
        print("snps must be one of: 'apoe', 'all_snps', 'LDE', 'polygenic_risk_score', 'none'")
        sys.exit()

    # which ages
    ddf = pd.read_parquet('allcausedementia.parquet', engine = 'fastparquet') # age, sex, all-patients
    if age_cutoff == True: # only patients 65 and older
        ddf = ddf[ddf['curr_age'] >= 65]

        df = df[df['IID'].isin(ddf['eid'])]
 
    # assign groups 
    if alzheimers_only: 
        df = pull_alzheimer_only(df, ddf)
    else:
        df = pull_all_dementia(df, ddf)

    drop_cols = ['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE', 'groups']
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df['groups'].values
    
    return X, y

def get_dir_path(experiment, age_cutoff, snps, alzheimers_only, model):  
    root = "./results_all"
    
    age_cond = "65up" if age_cutoff else "allages"
    alz_cond = "AD" if alzheimers_only else "ACD"

    path = f"{root}/{experiment}/{snps}/{age_cond}/{alz_cond}/{model}"  

    return path

def get_lancet_vars():
    """
    Returns two lists of variables related to a study.

    The first list, `lancet_vars`, contains a mix of categorical and continuous variables.
    The second list, `continuous_lancet_vars`, contains only continuous variables.

    Returns:
        tuple: A tuple containing two lists:
            - lancet_vars (list of str): A list of variable identifiers and names.
            - continuous_lancet_vars (list of str): A list of continuous variable identifiers.
    """
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
    continuous_lancet_vars = [
        "4700-0.0",
        "5901-0.0",
        "30780-0.0",
        "22038-0.0",
        "20161-0.0",
        "24012-0.0",
        "24018-0.0",
        "24019-0.0",
        "24006-0.0",
        "24015-0.0",
        "24011-0.0",
    ]
    return lancet_vars, continuous_lancet_vars

def subset_experiment_vars(X, experiment, lancet_vars):
    lancets = pd.read_parquet('../../../randy/rfb/tidy_data/UKBiobank/dementia/lancet2024/lancet2024_preprocessed.parquet', engine = 'fastparquet')
    demographics = pd.read_parquet('demographics.parquet', engine='fastparquet')

    # experiment variables
    experiment_vars = {
        "none": [],
        "age_alone": ['curr_age'],
        "all_demographics": 
            [
                "curr_age",
                "31-0.0", # 'max_educ_complete'
                "845-0.0",
                "21000-0.0",
            ],
        "age_sex_lancet2024":  
            [
                "curr_age",
                "31-0.0",
            ],
        "demographics_lancet2024":
            [
                "curr_age",
                "31-0.0",
                "845-0.0",
                "21000-0.0",
            ]   
    }

    features_to_add = experiment_vars[experiment]

    # Handle "none" experiment - only use SNP data, no demographics
    if experiment == "none":
        X = X.drop(columns=['IID'])
        print(f"None experiment: Using only SNP data")
        print(f"X shape: {X.shape}")
        print(f"X columns: {X.columns.tolist()}")
        return X

    # For all other experiments, add demographic features
    demo_sub = demographics.loc[:, ['eid'] + features_to_add]    
    
    # If snps is 'polygenic_risk_score', add 'polygenic_risk_score' to features_to_add
    if 'snps' in X.columns and X['snps'].iloc[0] == 'polygenic_risk_score':
        features_to_add = features_to_add + ['polygenic_risk_score']

    X = X.merge(demo_sub, right_on='eid', left_on='IID', how = 'left')
    X = X.drop(columns=['eid'])  # drop eid column after merge

    # subset lancet_variables
    if experiment in ("age_sex_lancet2024", "demographics_lancet2024"):
        lancet_cols = [c for c in lancet_vars if c in lancets.columns]
        lanc_sub = lancets.loc[:, ['eid'] + lancet_cols]
        X = X.merge(lanc_sub, right_on='eid', left_on='IID', how='left')
        X = X.drop(columns=['eid'])  # drop eid column after merge
    
    X = X.drop(columns=['IID'])

    print(X.shape)
    print(X.head())
    print(X.columns)
    
    return X

def continuous_vars_for_scaling(
    experiment, continuous_lancet_vars
):
    # choose the variables for scaling

    continuous_cols = {
        "none": [],
        "age_alone": ['curr_age'],
        "all_demographics": ['curr_age', "845-0.0"], 
        "age_sex_lancet2024": ['curr_age'] + continuous_lancet_vars,  # removed "845-0.0"
        "demographics_lancet2024": ['curr_age', "845-0.0"] + continuous_lancet_vars,
    }

    if experiment in continuous_cols:
        continuous_cols = continuous_cols[experiment]
    else:
        # output an error saying experiment is not in continuous_cols
        print("Experiment not in continuous_cols")
        sys.exit()

    return continuous_cols


def scale_continuous_vars(X_train, X_test, continuous_cols):
    """
    Scales the continuous variables in the training and test datasets using StandardScaler.
    Used only with lrl1 model.
    Parameters:
    X_train (pd.DataFrame): The training dataset.
    X_test (pd.DataFrame): The test dataset.
    continuous_cols (list of str): List of column names corresponding to continuous variables to be scaled.
    Returns:
    tuple: A tuple containing the scaled training and test datasets (X_train, X_test).
    """
    
    # Check if there are any continuous columns to scale
    if not continuous_cols:
        print("No continuous columns to scale")
        return X_train, X_test
    
    # Filter continuous_cols to only include columns that actually exist in the data
    existing_continuous_cols = [col for col in continuous_cols if col in X_train.columns]
    
    if not existing_continuous_cols:
        print(f"None of the specified continuous columns {continuous_cols} exist in the data")
        print(f"Available columns: {X_train.columns.tolist()}")
        return X_train, X_test

    print(f"Scaling continuous columns: {existing_continuous_cols}")
    
    scaler = StandardScaler()

    # Fit and transform only the existing continuous columns
    scaler.fit(X_train[existing_continuous_cols])
    X_train.loc[:, existing_continuous_cols] = scaler.transform(X_train[existing_continuous_cols])
    X_test.loc[:, existing_continuous_cols] = scaler.transform(X_test[existing_continuous_cols])

    return X_train, X_test

def subset_train_test(X, y, results_dir, fold_index):
    # logging
    logging.basicConfig(filename=f"{results_dir}/logging.txt", level=logging.INFO)
    logging.info(
            "Running experiment %s with IDP index %d and fold index %d",
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
    
def main(): 
    # parse the arguments
    (   
        fold_index,
        experiment,
        age_cutoff,
        snps, 
        alzheimers_only,
        model
    ) = parse_args()
    
    X, y = load_datasets(snps, age_cutoff, alzheimers_only)

    # create results dir
    results_dir = get_dir_path(experiment, age_cutoff, snps, alzheimers_only, model) + f"/{fold_index}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # choosing experimental variables
    print('subsetting experiment variables')
    lancet_vars, continuous_lancet_vars = get_lancet_vars() 
    X = subset_experiment_vars(X, experiment, lancet_vars)
    
    # Only get continuous columns if using lrl1 model
    if model == "lrl1":
        print("Scaling data for lrl1 classifier")
        continuous_cols = continuous_vars_for_scaling(
            experiment, continuous_lancet_vars
        )
        print(f"Continuous columns to scale: {continuous_cols}")

    # split + train
    print('splitting')
    X_train, y_train, X_test, y_test = subset_train_test(X, y, results_dir, fold_index)
    
    # Only scale if using lrl1 model AND there are continuous columns
    if model == "lrl1":
        X_train, X_test = scale_continuous_vars(X_train, X_test, continuous_cols)

    print('training')
    automl = AutoML()
    automl_settings = settings_automl(FLAML_TIME_BUDGET, metric="log_loss", model=model)
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
    # parse_args()
    # main() # this is the main function that runs the experiment