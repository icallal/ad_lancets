import argparse
import logging
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxTimeVaryingFitter
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

import pandas as pd
from datetime import datetime

from doubleml_utils import subset_train_test

def parse_args(): 

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_index", type=int, required=True)

    args = parser.parse_args()

    fold_index = args.fold_index

    return fold_index



# def build_ctv_from_diagnoses(dates_df, times_df, event_col='ad_dx'):
#     all_ctv = []

#     for eid, dx_row in dates_df.set_index('eid').iterrows():
#         time_row = times_df.set_index('eid').loc[eid]

#         start_time = time_row['start_time']
#         end_time = time_row['end_time']
#         ad_dx_date = dx_row[event_col]  # datetime of AD diagnosis (or NaT)

#         time_points = [start_time]
#         for col in dx_row.index:
#             if col == event_col:
#                 continue  # skip event_col for state tracking
#             date = dx_row[col]
#             if pd.notna(date) and date >= start_time:
#                 time_points.append(date)
#         if pd.notna(ad_dx_date) and ad_dx_date >= start_time:
#             time_points.append(ad_dx_date)
#         time_points.append(end_time)
#         time_points = sorted(set(time_points))

#         current_state = {col: 0 for col in dx_row.index if col != event_col}

#         for i in range(len(time_points) - 1):
#             t_start = time_points[i]
#             t_end = time_points[i+1]

#             if t_end == t_start:
#                 continue

#             for col in current_state:
#                 if dx_row[col] == t_start:
#                     current_state[col] = 1

#             # determine if event occurred in this interval (ad_dx = 1)
#             ad_dx = 1 if pd.notna(ad_dx_date) and ad_dx_date == t_end else 0

#             interval = {
#                 'eid': eid,
#                 'start_time': t_start,
#                 'end_time': t_end,
#                 **current_state,
#                 event_col: ad_dx
#             }
#             all_ctv.append(interval)

#     return pd.DataFrame(all_ctv)        

# #ctv = pd.read_parquet('lancet_dx_dates_ctv_encoded.parquet', engine = 'fastparquet')
# dx = pd.read_parquet('./snp_parquets/lancet_dx_dates.parquet', engine = 'fastparquet')

# for col in dx.drop(columns=['eid']).columns:
#     dx[col] = pd.to_datetime(dx[col], errors='coerce')

# demographics = pd.read_parquet('demographics.parquet', engine = 'fastparquet', columns = ['eid', '53-0.0']) # eid

# times_df = pd.DataFrame({
#     'eid': demographics['eid'],
#     'start_time': demographics['53-0.0'], 
#     'end_time': datetime(2023,1,1),
# })

# dx = build_ctv_from_diagnoses(dx, times_df)

# dx.to_parquet('lancet_ctv_encoded.parquet', engine = 'fastparquet')
# print('ctv encoded!')


# read in dfs
df = pd.read_parquet('../../../../randy/proj_idp/tidy_data/prs_Alz/prs_Alz.parquet', engine = 'fastparquet') # eid
#df = df.drop(columns = ['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], errors = 'ignore')

ctv = pd.read_parquet('lancet_ctv_encoded.parquet', engine = 'fastparquet')

df = ctv.merge(df, on = 'eid', how='left')

# add covariates
covariates = ['31-0.0', 'curr_age', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4', 'education_years', 'e2/e2']

covs = pd.read_parquet('doubleML_covariates.parquet', engine='fastparquet')
covs.rename(columns={'IID':'eid'}, inplace=True)
covs = covs[['eid'] + covariates]

df = df.merge(covs, on='eid', how='inner')

df = df.dropna()

fold_index = parse_args()

# set results dir
results_dir = './results_survival/ctv_lancets/prs' + f"/{fold_index}"
if not os.path.exists(results_dir):
        os.makedirs(results_dir)

# split
print('splitting')
X_train, _, _, _ = subset_train_test(df, df['ad_dx'], results_dir, fold_index)

 # Remove duplicate columns
X_train = X_train.loc[:, ~X_train.T.duplicated()]

# Remove columns with zero variance
nunique = X_train.nunique()
X_train = X_train.loc[:, nunique > 1]

# Remove perfectly correlated columns
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] == 1)]
X_train = X_train.drop(columns=to_drop)

# =========================
# Cox Time-Varying Fitting
# =========================
print('fitting')
ctv = CoxTimeVaryingFitter(penalizer=0.1)  # add slight ridge penalty

ctv.fit(
    X_train,
    id_col='eid',
    start_col='start_time',
    stop_col='end_time',
    event_col='ad_dx',
)

results = ctv.summary
results.to_csv(f'{results_dir}/ctv_results')
joblib.dump(ctv, f"{results_dir}/ctv_model.joblib")