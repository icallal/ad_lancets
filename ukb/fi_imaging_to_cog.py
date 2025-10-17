import pandas as pd
from lightgbm import LGBMRegressor
from cates_utils import covariates_to_add_cates, load_cates_3d_array

overlap_cog = ['4282-2.0', '20016-2.0', '20023-2.0', '20197-2.0']
covariates_df = pd.read_parquet('doubleML_covariates.parquet', engine='fastparquet', 
                               columns=['IID', 'bmi', 'mdi', 'education_years', 'curr_age', '31-0.0', 
                                      'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4', 'groups'])

covariates_df.dropna(subset = ['bmi', 'curr_age', '31-0.0', 'groups', 'education_years', 'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4'], inplace=True)

covariates_df.rename(columns={'IID': 'eid'}, inplace=True)
ct_df = pd.read_parquet('cognitive_test_results_2.parquet', engine='fastparquet', 
                       columns=['eid'] + overlap_cog)

imaging_df = pd.read_parquet('../../../randy/rfb/tidy_data/UKBiobank/dementia/neuroimaging/X.parquet', engine='fastparquet')
imaging_only = imaging_df[['eid'] + [col for col in imaging_df.columns if (col.startswith('25') or col.startswith('27') or col.startswith('26'))]]

covariates_df = covariates_df.merge(imaging_only, on='eid', how='inner').merge(ct_df, on='eid', how='inner')

cates_array, imaging_metrics, cognitive_tests, n_patients = load_cates_3d_array('double_ml/imaging/results/imaging_to_cog/cates/cates_3d_array.npz')
test_df = covariates_to_add_cates('4282-2.0', '25000-2.0', covariates_df, cates_array, imaging_metrics, cognitive_tests)

feature_importances = {}

for i in imaging_metrics: 
    feature_importances[i] = {}
    for c in cognitive_tests:
        test_df = covariates_to_add_cates(c, i, covariates_df, cates_array, imaging_metrics, cognitive_tests)

        # fit
        X = test_df.drop(columns=['cates_value', 'eid', c, i])
        y = test_df['cates_value']

        LGBM_model = LGBMRegressor()
        LGBM_model.fit(X, y)

        # construct feature importances table
        fi = LGBM_model.feature_importances_
        fnames = LGBM_model.feature_name_

        feature_importance_df = pd.DataFrame({'feature': fnames, 'importance': fi})
        feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

        feature_importances[i][c] = feature_importance_df


