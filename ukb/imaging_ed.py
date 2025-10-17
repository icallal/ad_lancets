from doubleml_utils import run_dml, summarize_results

import pandas as pd


covariates_df = pd.read_parquet('doubleML_dep_AD_covariates.parquet', engine = 'fastparquet')
imaging_df = pd.read_parquet('../../../randy/rfb/tidy_data/UKBiobank/dementia/neuroimaging/X.parquet', engine = 'fastparquet')

covariates_df.rename(columns = {'IID': 'eid'}, inplace = True)
imaging_only = imaging_df[['eid'] + [col for col in imaging_df.columns if (col.startswith('25') or col.startswith('27') or col.startswith('26'))]]

covariates_df = covariates_df.merge(imaging_only, on='eid', how='inner')

# ed to image
# image to AD
results = {}

# Impute education years ONCE
covariates_df['education_years'] = covariates_df['education_years'].fillna(
    covariates_df['education_years'].mean()
)
    
for i in imaging_only.drop(columns=['eid']).columns:
    # Create boolean mask for non-missing values
    valid_mask = ~covariates_df[i].isna()
    
    # Check sample size before proceeding
    n_valid = valid_mask.sum()
    if n_valid < 100:
        print(f"Skipping {i} due to insufficient data ({n_valid} samples)")
        continue

    # Use boolean indexing - no copying!
    outcome = covariates_df.loc[valid_mask, i]
    exposure = covariates_df.loc[valid_mask, 'education_years']
    covariates = covariates_df.loc[valid_mask, ['curr_age', '31-0.0', 'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4']]

    try:
        result = run_dml(covariates, outcome, exposure)
        results[i] = result.summary
        print(f"Completed {i}: {n_valid} samples")
    except Exception as e:
        print(f"Error processing {i}: {str(e)}")
        continue


results = summarize_results(results)
results.to_parquet('imaging_ed_to_image.parquet', engine = 'fastparquet')
