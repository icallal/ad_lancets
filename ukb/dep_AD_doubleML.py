from fastparquet import ParquetFile
import pandas as pd
from ml_exp import encode_education
from ml_exp import pull_alzheimer_only
import numpy as np

import doubleml as dml
from doubleml.data import DoubleMLData
from doubleml import DoubleMLPLR
from lightgbm import LGBMRegressor
import statsmodels.api as sm 

import os
import joblib

covariates_df = pd.read_parquet('doubleML_dep_AD_covariates.parquet', engine = 'fastparquet')

# depression snps
loaded = np.load('depression.npz', allow_pickle=True)
data = loaded['data']
columns = loaded['columns']
depression_df = pd.DataFrame(data, columns=columns)
depression_df = depression_df.drop(columns = ['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'])

covariates_df = covariates_df.merge(depression_df, on='IID', how = 'inner')

# Setup: outcome y, covariates x, treatment d
covariates = covariates_df.drop(columns = ['groups', 'depression', 'IID'])
outcome = covariates_df['groups']
exposure = covariates_df['depression']

covariates = covariates.fillna(covariates.mean())

data = DoubleMLData.from_arrays(x=covariates, y=outcome, d=exposure)  

dml_model = DoubleMLPLR(data, 
                        ml_l=LGBMRegressor(),  # Y|X (binary outcome)
                        ml_m=LGBMRegressor(),  # D|X
                        n_folds=5)
dml_model.fit(store_models=True)

joblib.dump(dml_model, './double_ml/depression_AD_snps_regressed.joblib')

# get residuals
AD_pred = dml_model.predictions['ml_l'][:,0,0].ravel()
dep_pred = dml_model.predictions['ml_m'][:,0,0].ravel()
AD_hat = outcome - AD_pred
dep_hat = exposure - dep_pred

# linear model with snps
X = pd.concat([dep_hat, depression_df], axis = 1)
X = sm.add_constant(X)

X = X.fillna(0)  # or X = X.fillna(X.mean())

X, AD_hat_aligned = X.align(AD_hat, join='inner', axis=0)

model = sm.OLS(AD_hat, X).fit()
print(model.summary())

# Save to joblib
os.makedirs('./double_ml', exist_ok=True)
joblib.dump(model, './double_ml/linear_snps_regressed.joblib')