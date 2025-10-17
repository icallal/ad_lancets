
# the dfs to load

import pandas as pd
from datetime import datetime

from lifelines import CoxTimeVaryingFitter
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

def build_ctv_from_diagnoses(dates_df, times_df, event_col='ad_dx'):
    all_ctv = []

    for eid, dx_row in dates_df.set_index('eid').iterrows():
        time_row = times_df.set_index('eid').loc[eid]

        start_time = time_row['start_time']
        end_time = time_row['end_time']
        ad_dx_date = dx_row[event_col]  # datetime of AD diagnosis (or NaT)

        time_points = [start_time]
        for col in dx_row.index:
            if col == event_col:
                continue  # skip event_col for state tracking
            date = dx_row[col]
            if pd.notna(date) and date >= start_time:
                time_points.append(date)
        if pd.notna(ad_dx_date) and ad_dx_date >= start_time:
            time_points.append(ad_dx_date)
        time_points.append(end_time)
        time_points = sorted(set(time_points))

        current_state = {col: 0 for col in dx_row.index if col != event_col}

        for i in range(len(time_points) - 1):
            t_start = time_points[i]
            t_end = time_points[i+1]

            if t_end == t_start:
                continue

            for col in current_state:
                if dx_row[col] == t_start:
                    current_state[col] = 1

            # determine if event occurred in this interval (ad_dx = 1)
            ad_dx = 1 if pd.notna(ad_dx_date) and ad_dx_date == t_end else 0

            interval = {
                'eid': eid,
                'start_time': t_start,
                'end_time': t_end,
                **current_state,
                event_col: ad_dx
            }
            all_ctv.append(interval)

    return pd.DataFrame(all_ctv)

def rebase_times(df):
    df = df.copy()
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    min_start_times = df.groupby('IID')['start_time'].transform('min')

    df['start_time_days'] = (df['start_time'] - min_start_times).dt.days  
    df['end_time_days'] = (df['end_time'] - min_start_times).dt.days

    df = df.drop(columns=['start_time', 'end_time'])
    df = df.rename(columns={'start_time_days': 'start_time', 'end_time_days': 'end_time'})

    return df


ctv = pd.read_parquet('lancet_dx_dates_ctv_encoded.parquet', engine = 'fastparquet')
dx = pd.read_parquet('lancet_dx_dates.parquet', engine = 'fastparquet')

demographics = pd.read_parquet('demographics.parquet', engine = 'fastparquet', columns = ['eid', '53-0.0']) # eid

times_df = pd.DataFrame({
    'eid': demographics['eid'],
    'start_time': demographics['53-0.0'], 
    'end_time':   datetime(2023,1,1),
})


ctv = build_ctv_from_diagnoses(dx, times_df)

ctv.to_parquet('lancet_dx_dates_ints_encoded.parquet', engine = 'fastparquet')

ctv = rebase_times(ctv)

# load in covariates
covariates_df = pd.read_parquet('doubleML_dep_AD_covariates.parquet', engine = 'fastparquet')

covariates_df['30710-0.0'] = covariates_df['30710-0.0'].fillna(covariates_df['30710-0.0'].mean())
covariates_df['mdi'] = covariates_df['mdi'].fillna(covariates_df['mdi'].mean())

covariates_df = covariates_df.drop(columns = 'groups')

# merge with ctv
ctv = ctv.merge(covariates_df, on = 'IID', how = 'inner')
ctv = ctv.drop(columns = ['depression', 'curr_age'])


X_train, _, _, _ = train_test_split(ctv, ctv['groups'], test_size = 0.2, random_state=1928)

ctv_model = CoxTimeVaryingFitter(penalizer=0.1)  # add slight ridge penalty

ctv_model.fit(
    X_train,
    id_col='IID',
    start_col='start_time',
    stop_col='end_time',
    event_col='groups',
)

ctv_model.print_summary()
ctv_model.save_model(f"./results_survival/ctv_model_correct.pkl")