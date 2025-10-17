import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def rename_vars(): 
    rename = {
            "4700-0.0": 'Age Cataract Diagnosed',
            "5901-0.0": 'Age Diabetic Retinopathy Diagnosed',
            "30780-0.0": 'LDL',
            "head_injury": 'Head Injury',
            "22038-0.0": 'Min/Week Moderate Activity',
            "20161-0.0": 'Years of Smoking',
            "alcohol_consumption": 'Alcohol consumption',
            "hypertension": 'Hypertension',
            "obesity": 'Obesity',
            "diabetes": 'Diabetes',
            "hearing_loss": 'Hearing Loss',
            "depression": 'Depression',
            "freq_friends_family_visit": 'Frequency of Friends/Family Visits',
            "24012-0.0": 'Distance to Major Road',
            "24018-0.0": 'NO2 Air Pollution',
            "24019-0.0": 'PM10 Air Pollution',
            "24006-0.0": 'PM2.5 Air Pollution',
            "24015-0.0": 'Amount of Major Roads',
            "24011-0.0": 'Traffic Intensity',
            '6138-0.0': 'Education Level',
            '845-0.0': 'Age Completed Education',
            'curr_age': 'Current Age',
    }
    return rename

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
        logging.info(f"Processing fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, y_train, X_test, y_test

def scale_continuous_vars(df, continuous_cols):
    """
    Scales the continuous variables in the dataframe using StandardScaler. Returns scaled df. 
    """

    scaler = StandardScaler()

    # Fit and transform only the continuous columns
    scaler.fit(df[continuous_cols])
    df.loc[:, continuous_cols] = scaler.transform(df[continuous_cols])
   
    return df

def encode(df, col, prefix):
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

import doubleml as dml
from doubleml import DoubleMLPLR, DoubleMLPLIV
from doubleml import DoubleMLData
from lightgbm import LGBMClassifier, LGBMRegressor

def run_dml(covariates, outcome, exposure):
    data = DoubleMLData.from_arrays(x=covariates, y=outcome, d=exposure)  

    dml_model = DoubleMLPLR(data, 
                            ml_l=LGBMRegressor(),  # Y|X (binary outcome)
                            ml_m=LGBMRegressor(),  # D|X
                            n_folds=5)
    dml_model.fit(store_models=True)

    return dml_model

def run_dml_instrument(df, covariates, outcome, exposure, instrument):
    
    ml_g = LGBMRegressor() # outcome, instrument
    ml_m = LGBMRegressor() # confounders, instrument
    ml_r = LGBMRegressor() # decision, instrument

    obj_dml_data = dml.DoubleMLData(
        df, y_col=outcome, d_cols=exposure,
        z_cols=instrument, x_cols=covariates
    )

    dml_model = DoubleMLPLIV(obj_dml_data, ml_g, ml_m, ml_r)
    dml_model.fit()

    return dml_model

# TOOLS for summarizing results

def merge_folds(exp, path):
    path = f'{path}/{exp}/'
    all_results = []

    for fname in os.listdir(path):
        if fname.startswith('imaging_to_cogtest_fold_') and fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, fname), sep=None, engine='python')
            all_results.append(df)

    massive_df = pd.concat(all_results, ignore_index=True)
    #massive_df = massive_df.drop_duplicates(subset=['index', 'protein'])

    massive_df.to_csv(f'{path}/all_results.csv', index=False)

    return massive_df

def summarize_results(results): 
    rows = []

    for test_id, df in results.items():
        # Pull the row for `d` as a dictionary and tag with the test_id
        row = df.loc['d'].to_dict()
        row['test_id'] = test_id
        rows.append(row)

    # Convert to a DataFrame
    summary_df = pd.DataFrame(rows)

    # Move 'test_id' to the front
    summary_df = summary_df[['test_id'] + [col for col in summary_df.columns if col != 'test_id']]

    return summary_df

def flatten_imaging_cog_dict(data):
    """
    Flattens a nested dict of the form:
    { imaging_metric: { cognitive_test: DataFrame, ... }, ... }
    into a long-format DataFrame.
    """
    records = []
    for imaging_metric, cog_tests in data.items():
        for cog_test, df in cog_tests.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                row = df.iloc[0].to_dict()
                row['Cognitive Test'] = cog_test
                row['Imaging Metric'] = imaging_metric
                records.append(row)
    return pd.DataFrame(records)[[
        'Imaging Metric', 'Cognitive Test', 'coef', 'std err', 't', 'P>|t|', '2.5 %', '97.5 %'
    ]]