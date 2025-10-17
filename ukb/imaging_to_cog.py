from doubleml_utils import run_dml_instrument, run_dml 
import pandas as pd
import argparse
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from econml.iv.dml import DMLIV
from lightgbm import LGBMRegressor

from cates_utils import cates_to_3d_array, save_cates_3d_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    return args.fold

fold = parse_args()

overlap_cog = ['4282-2.0', '20016-2.0', '20023-2.0', '20197-2.0']
covariates_df = pd.read_parquet('doubleML_covariates.parquet', engine='fastparquet', 
                               columns=['IID', 'bmi', 'mdi', 'education_years', 'curr_age', '31-0.0', 
                                      'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4', 'groups'])
covariates_df.rename(columns={'IID': 'eid'}, inplace=True)
ct_df = pd.read_parquet('cognitive_test_results_2.parquet', engine='fastparquet', 
                       columns=['eid'] + overlap_cog)

imaging_df = pd.read_parquet('../../../randy/rfb/tidy_data/UKBiobank/dementia/neuroimaging/X.parquet', engine='fastparquet')
imaging_only = imaging_df[['eid'] + [col for col in imaging_df.columns if (col.startswith('25') or col.startswith('27') or col.startswith('26'))]]

covariates_df = covariates_df.merge(imaging_only, on='eid', how='inner').merge(ct_df, on='eid', how='inner')

results = {}
cates = {}

covariates_df.dropna(subset = ['bmi', 'curr_age', '31-0.0', 'groups', 'education_years', 'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4'], inplace=True)
covariates_df['age_squared'] = covariates_df['curr_age'] ** 2

#sig_results['Imaging Metric'].unique().tolist()

imaging_columns = imaging_only.drop(columns=['eid']).columns.tolist()
num_imaging_cols = len(imaging_columns)

startidx = fold * 50
endidx = min(50 * (fold + 1), num_imaging_cols)

for i in imaging_only.drop(columns=['eid']).columns.tolist()[startidx:endidx]:
    print(f"Processing imaging metric: {i}")
    valid_mask = ~covariates_df[i].isna()
    covariates_df = covariates_df[valid_mask]

    results[i] = {}
    cates[i] = {}

    for test in ['4282-2.0', '20016-2.0', '20023-2.0', '20197-2.0']:
        print('starting test')

        valid_mask = ~covariates_df[test].isna()
        covariates_df = covariates_df[valid_mask]

        Y = covariates_df[test]
        T = covariates_df[i]
        Z = covariates_df['31-0.0']
        if i not in ['25001-2.0', '25003-2.0', '25005-2.0', '25007-2.0', '25009-2.0']:  
            X = covariates_df[['bmi', 'mdi', 'curr_age', 'age_squared', 'e2/e2', 'e3/e3', 
                            'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4', 'education_years', '25000-2.0', 'age_squared']]
        else: 
            X = covariates_df[['bmi', 'mdi', 'curr_age', 'age_squared', 'e2/e2', 'e3/e3', 
                            'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4', 'education_years', 'age_squared']]

        print(Y.isna().sum(), T.isna().sum(), X.isna().sum(), Z.isna().sum())

        print(f"Fitting model for {i} and {test} with {len(Y)} samples")
        # DML with forest-based CATE estimator
        model = DMLIV(
            model_y_xw=LGBMRegressor(),
            model_t_xw=LGBMRegressor(),
            model_t_xwz=LGBMRegressor(),
            discrete_treatment=False,
            discrete_instrument=True,
            random_state=42
        )
        model.fit(Y, T, Z=Z, X=X)

        # Estimate overall ATE
        ate = model.ate(X)
        results[i][test] = {"ATE": ate, "model": model, "summary": model.summary}

        # Estimate CATEs
        cate = model.effect(X)
        
        cates[i][test] = cate

def cates_to_dataframe(cates_dict):
    """
    Convert nested dict of CATEs to a DataFrame.
    Structure: {imaging_metric: {cognitive_test: array, ...}, ...}
    """
    all_data = []
    
    for imaging_metric, cog_tests in cates_dict.items():
        for cog_test, cate_array in cog_tests.items():
            # Create a DataFrame for this combination
            temp_df = pd.DataFrame({
                'imaging_metric': imaging_metric,
                'cognitive_test': cog_test,
                'cate_value': cate_array,
                'subject_id': range(len(cate_array))  # Add subject identifier
            })
            all_data.append(temp_df)
    
    # Concatenate all DataFrames
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# Convert your cates to DataFrame
cates_df = cates_to_dataframe(cates)
print(f"Shape: {cates_df.shape}")
print(cates_df.head(10))

# Convert your cates dictionary
cates_df = cates_to_dataframe(cates)

# Save it
cates_df.to_csv('./double_ml/imaging/results/imaging_to_cog/cates_headsz_regressed/cates_individual_effects_fold_{fold}.csv', index=False)

# array_3d, imaging_metrics, cognitive_tests, n_patients = cates_to_3d_array(cates)
# save_cates_3d_array(array_3d, imaging_metrics, cognitive_tests, n_patients, f'double_ml/imaging/results/imaging_to_cog_npz/cates_npz/cates_imaging_to_cogtest_fold_{fold}.npz')

# # Impute education years ONCE
# covariates_df.dropna(subset=['education_years'], inplace=True)

# # CORRECTED: Calculate indices based on number of imaging columns
# imaging_columns = imaging_only.drop(columns=['eid']).columns.tolist()
# num_imaging_cols = len(imaging_columns)

# startidx = fold * 50
# endidx = min(50 * (fold + 1), num_imaging_cols)

# print(f"Processing fold {fold}: column indices {startidx} to {endidx}")

# for i in imaging_columns[startidx:endidx]:
#     valid_mask = ~covariates_df[i].isna()
#     covariates_df = covariates_df[valid_mask]

#     results[i] = {}

#     for c in ['4282-2.0', '20016-2.0', '20023-2.0', '20197-2.0']:
#     # Create boolean mask for non-missing values
#         valid_mask = ~covariates_df[c].isna()
        
#         # Check sample size before proceeding
#         n_valid = valid_mask.sum()
#         if n_valid < 100:
#             print(f"Skipping {i} due to insufficient data ({n_valid} samples)")
#             continue

#         # Use boolean indexing - no copying!
#         covariates_df = covariates_df[valid_mask]

#         outcome = c
#         exposure = i
#         covariates = ['curr_age', 'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4']
#         instrument = '31-0.0'

#         try:
#             result = run_dml_instrument(covariates_df, covariates=covariates, outcome=outcome, exposure=exposure, instrument=instrument)
#             results[i][c] = result.summary
#             print(f"Completed {i}: {n_valid} samples")
#         except Exception as e:
#             print(f"Error processing {i}: {str(e)}")
#             continue

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

final_df = flatten_imaging_cog_dict(results)
final_df.to_csv(f'double_ml/imaging/results/imaging_to_cog/imaging_to_cogtest_fold_{fold}.csv', index=False)
