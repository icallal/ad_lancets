import argparse
import pandas as pd
import os
from datetime import datetime
import numpy as np
from ml_exp import pull_alzheimer_only
import joblib
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from doubleml_utils import subset_train_test

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_index", type=int, required=True)

    args = parser.parse_args()

    fold_index = args.fold_index

    return fold_index

def encode_start_end_CPH(feature, times_df, df):

    # 1) load the raw dates for this feature
    diagnosis = df[['IID', feature]]
    #diagnosis = pd.read_parquet('../../../randy/rfb/tidy_data/UKBiobank/dementia/lancet2024/lancet2024_preprocessed.parquet', engine = 'fastparquet', columns = ['eid', feature])

    # 2) ensure your times_df bounds are datetime
    times_df = times_df.copy()
    times_df['start_time'] = pd.to_datetime(times_df['start_time'])
    times_df['end_time']   = pd.to_datetime(times_df['end_time'])

    # 3) join on IID to get start/end for every person
    # Ensure diagnosis date is in datetime
    diagnosis = diagnosis.rename(columns={'IID':'eid'})
    diagnosis = diagnosis.merge(times_df, on='eid', how='inner')
    diagnosis[feature] = pd.to_datetime(diagnosis[feature], errors='coerce')

    # 4) build the interval rows
    intervals = []
    for _, row in diagnosis.iterrows():
        eid   = row['eid']
        start = row['start_time']
        end   = row['end_time']
        dx    = row[feature] # date of diagnosis

        if pd.isna(dx) or dx > end:
            # never diagnosed within window
            intervals.append({
                'eid': eid,
                'start_time': start,
                'end_time': end,
                'diagnosis': 0
            })
        else:
            intervals.append({
                'eid': eid,
                'start_time': start,
                'end_time': dx,
                'diagnosis': 1
            })

    encoded = pd.DataFrame(intervals)

    # 5) pull in every other column from df (snps, duration, groups, etc.)
    #    drop start_time/end_time there so we don’t overwrite our intervals
    meta = df.drop(columns=['start_time','end_time'], errors='ignore')
    encoded = encoded.rename(columns={'eid': 'IID'})
    encoded = encoded.merge(meta, on='IID', how='left')

    return encoded

lancets = pd.read_parquet('../../../../randy/rfb/tidy_data/UKBiobank/dementia/lancet2024/lancet2024_preprocessed.parquet', engine = 'fastparquet')

# Extract the left-hand values (column names) from the commented lines
lancet_cols = [
    'eid',
    'hypertension',
    'obesity',
    'diabetes',
    'hearing_loss',
    'head_injury',
    'freq_friends_family_visit',
    'depression',
    'alcohol_consumption',
    '5901-0.0',
    '4700-0.0',
    '1558-0.0',
    '2020-0.0',
    '1031-0.0',
    '24018-0.0',
    '24011-0.0',
    '24015-0.0',
    '24012-0.0',
    '24019-0.0',
    '30780-0.0',
    '24006-0.0'
]
lancets = lancets[lancet_cols].dropna()


def main():
    fold_index = parse_args()

    # df = pd.read_parquet('./snp_parquets/raw_allsnps_AD.parquet', engine = 'fastparquet') # IID
    ddf = pd.read_parquet('allcausedementia.parquet', engine = 'fastparquet') # eid

    df = pd.read_parquet('../../../../randy/rfb/tidy_data/UKBiobank/dementia/lancet2024/lancet2024_preprocessed.parquet', engine = 'fastparquet', 
                        columns = lancet_cols)

    demographics = pd.read_parquet('demographics.parquet', engine = 'fastparquet', columns = ['eid', '53-0.0']) # eid
    demographics['53-0.0'] = pd.to_datetime(demographics['53-0.0'], errors='coerce')
    demographics = demographics.dropna(subset=['53-0.0'])

    # add start and end times to df
    times_df = pd.DataFrame({
        'eid': df['eid'],
        'start_time': demographics['53-0.0'], 
        'end_time':   datetime(2025,1,1),

    })

    df = df.merge(times_df, on ='eid', how='left')
    df = df.dropna(subset=['eid'])
    df = df.rename(columns={'eid': 'IID'})

    # add the covariates!!!!!!!!!!!!!
    covariates = pd.read_parquet('doubleML_covariates.parquet', engine = 'fastparquet', 
                                 columns=['IID', 'curr_age', '31-0.0', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4', 'e2/e2', 'mdi', 'education_years']) # eid
    df = df.merge(covariates, on='IID', how='left')
    df.dropna(subset=['education_years'], inplace=True)

    # encode duration until alzheimer's diagnosis
    df = pull_alzheimer_only(df, ddf) # assign
    df = df.merge(
        ddf[['eid', '131036-0.0']].rename(columns={'eid': 'IID', '131036-0.0': 'diagnosis_date'}),
        on ='IID',
        how='left'
    )

    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')

    # assign duration: if group==1, duration = diagnosis_date - start_time, else duration = end_time - start_time
    df = df.merge(ddf[['eid', '131036-0.0']], left_on='IID', right_on='eid', how='left')
    df['duration'] = np.where(
        df['groups'] == 1,
        (df['diagnosis_date'] - df['start_time']).dt.days,
        (df['end_time'] - df['start_time']).dt.days
    )
    df = df[df['duration'] >= 0].copy()  # filter out negative durations
    df = df.drop(columns=['131036-0.0'])

    # impute
    cols_to_drop = df.isna().mean()[df.isna().mean() > 0.1].index
    df = df.drop(columns=cols_to_drop)

    snp_cols = df.columns[df.columns.str.startswith('rs')]
    df[snp_cols] = df[snp_cols].fillna(0)

    # some randome cleaning and shit
    df = df.drop(columns=['FID', 'PAT', 'IID', 'MAT', 'SEX', 'PHENOTYPE', 'diagnosis_date', 'start_time', 'end_time'], errors = 'ignore') # diagnosis

    # to use with scikit survival, change groups 01 to true false
    df['groups'] = df['groups'].astype(bool)
    df['duration'] = df['duration'].astype(float)

    df = df.dropna()

    results_dir = f'./results_survival/cph_lancets_sksurv/{fold_index}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print('splitting')
    X_train, y_train, X_test, y_test = subset_train_test(df, df['groups'], results_dir, fold_index)

    y = Surv.from_dataframe("groups", "duration", X_train)
    X = X_train.drop(columns=["groups", "duration"], errors='ignore')

    coxph = CoxPHSurvivalAnalysis()

    print('fitting')
    coxph.fit(X, y)
    joblib.dump(coxph, f'{results_dir}/sksurv_model.joblib')

    # # read in dfs
    # df = pd.read_parquet('./snp_parquets/raw_allsnps_AD.parquet', engine = 'fastparquet') # IID

    # ddf = pd.read_parquet('allcausedementia.parquet', engine = 'fastparquet') # eid

    # demographics = pd.read_parquet('demographics.parquet', engine = 'fastparquet', columns = ['eid', '53-0.0']) # eid
    # demographics['53-0.0'] = pd.to_datetime(demographics['53-0.0'], errors='coerce')
    # demographics = demographics.dropna(subset=['53-0.0'])


    # # add start and end times to df
    # times_df = pd.DataFrame({
    #     'eid': ddf['eid'],
    #     'start_time': demographics['53-0.0'], 
    #     'end_time':   datetime(2025,1,1),
    # })

    # df = df.merge(times_df, left_on='IID', right_on ='eid', how='left')
    # df = df.dropna(subset=['eid'])
    # df = df.drop(columns=['eid'])

    # # encode duration until alzheimer's diagnosis
    # df = pull_alzheimer_only(df, ddf) # assign
    # df = df.merge(
    #     ddf[['eid', '131036-0.0']].rename(columns={'eid': 'IID', '131036-0.0': 'diagnosis_date'}),
    #     on='IID',
    #     how='left'
    # )

    # df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')

    # # assign duration: if group==1, duration = diagnosis_date - start_time, else duration = end_time - start_time
    # df = df.merge(ddf[['eid', '131036-0.0']], left_on='IID', right_on='eid', how='left')
    # df['duration'] = np.where(
    #     df['groups'] == 1,
    #     (df['diagnosis_date'] - df['start_time']).dt.days,
    #     (df['end_time'] - df['start_time']).dt.days
    # )
    # df = df[df['duration'] >= 0].copy()  # filter out negative durations
    # df = df.drop(columns=['131036-0.0'])

    # # merge on lancets
    # df = df.merge(lancets, left_on='IID', right_on='eid', how='inner')

    # # # impute
    # cols_to_drop = df.isna().mean()[df.isna().mean() > 0.1].index
    # df = df.drop(columns=cols_to_drop)

    # snp_cols = df.columns[df.columns.str.startswith('rs')]
    # df[snp_cols] = df[snp_cols].fillna(0)

    # # some randome cleaning and shit
    # df = df.drop(columns=['FID', 'PAT', 'IID', 'MAT', 'SEX', 'PHENOTYPE', 'diagnosis_date', 'start_time', 'end_time', 'eid'], errors = 'ignore') # diagnosis

    # df['groups'] = df['groups'].astype(bool)

    
    # # SPLITTING 
    # fold_index = parse_args()
    # results_dir = f'./results_survival/cph_lancets/{fold_index}'

    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    
    # print('splitting')
    # X_train, _, X_test, _ = subset_train_test(df, df['groups'], results_dir, fold_index)
    
    # # Store the columns that should be kept for prediction
    # target_cols = ['duration', 'groups']
    
    # # cleaning - Apply to BOTH train and test sets consistently
    # print(f"Before cleaning - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # # Remove duplicate columns (based on training data)
    # train_deduped = X_train.loc[:, ~X_train.T.duplicated()]
    # cols_to_keep = train_deduped.columns
    # X_train = X_train[cols_to_keep]
    # X_test = X_test[cols_to_keep]
    
    # # Remove columns with zero variance (based on training data)
    # nunique = X_train.nunique()
    # cols_with_variance = nunique[nunique > 1].index
    # X_train = X_train[cols_with_variance]
    # X_test = X_test[cols_with_variance]
    
    # # Remove perfectly correlated columns (based on training data)
    # corr_matrix = X_train.corr().abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # to_drop = [column for column in upper.columns if any(upper[column] == 1)]
    # X_train = X_train.drop(columns=to_drop)
    # X_test = X_test.drop(columns=to_drop)
    
    # # Remove columns with any NaNs (based on training data)
    # cols_no_nan = X_train.dropna(axis=1, how='any').columns
    # X_train = X_train[cols_no_nan]
    # X_test = X_test[cols_no_nan]
    
    # print(f"After cleaning - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # y = Surv.from_dataframe("groups", "duration", X_train)
    # X = X_train.drop(columns=target_cols, errors='ignore')
    # coxph = CoxPHSurvivalAnalysis()
    # coxph.fit(X, y)

    # # # Fit Cox model
    # # cph = CoxPHFitter()
    # # cph.fit(X_train, duration_col='duration', event_col='groups')
    
    # # Save model
    # model_path = f'{results_dir}/model_surv.joblib'
    # joblib.dump(coxph, model_path)

    # X_test.to_parquet(f'{results_dir}/X_test.parquet', engine = 'fastparquet')

if __name__ == "__main__":
    main()
