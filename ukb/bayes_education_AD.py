import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
import pytensor
#pytensor.config.cxx = '/usr/bin/clang++'


covariates_df = pd.read_parquet('doubleML_dep_AD_covariates.parquet', engine = 'fastparquet', columns=['IID', 'education_years', 'curr_age', 
                                                                                                       '31-0.0', 'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4', 'groups'])
covariates_df = covariates_df.dropna(subset=['education_years', 'curr_age', '31-0.0', 'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4'])

df = covariates_df.rename(columns={
    'curr_age': 'age',
    '31-0.0': 'sex',
    'education_years': 'education',
    'groups': 'AD_diagnosis'
})

# cut off at age 65
df = df[df['age'] >= 65]


apoe_cols = ['e2/e2', 'e2/e3', 'e2/e4', 'e3/e3', 'e3/e4', 'e4/e4']
reference_apoe = 'e3/e3'

# Standardize continuous variables for better sampling
df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['education_std'] = (df['education'] - df['education'].mean()) / df['education'].std()


print(f"Data shape: {df.shape}")
print(f"AD prevalence: {df['AD_diagnosis'].mean():.3f}")
print(f"Reference APOE genotype: {reference_apoe}")
print(f"APOE dummy columns: {apoe_cols}")
print(f"APOE genotype counts:")

# Bayesian logistic regression with interactions
# Only include interactions, not main effects of modifiers
with pm.Model() as model:
    intercept = pm.Normal('intercept', mu=0, sigma=2.5)
    beta_edu = pm.Normal('education', mu=0, sigma=1)
    
    # Only interaction terms - no main effects for age, sex, APOE
    beta_edu_age = pm.Normal('education_age', mu=0, sigma=0.5)
    beta_edu_sex = pm.Normal('education_sex', mu=0, sigma=0.5)
    beta_edu_apoe = pm.Normal('education_APOE', mu=0, sigma=0.5, shape=len(apoe_cols))
    
    mu = (intercept + 
          beta_edu * df['education_std'] +
          beta_edu_age * df['education_std'] * df['age_std'] +
          beta_edu_sex * df['education_std'] * df['sex'] +
          pm.math.dot(df[apoe_cols].values * df['education_std'].values.reshape(-1, 1), 
                     beta_edu_apoe))
    
    # Likelihood
    p = pm.Deterministic('p', pm.math.sigmoid(mu))
    y_obs = pm.Bernoulli('y_obs', p=p, observed=df['AD_diagnosis'])
    
    # Sample from posterior
    trace = pm.sample(3000, tune=2000, chains=4)


# Check convergence
print(az.summary(trace, var_names=['intercept', 'education', 
                                   'education_age', 'education_sex', 'education_APOE']))

# Calculate CATE estimates for each individual
def calculate_cate_posterior(trace, df):
    """Calculate CATE estimates with uncertainty for each individual"""
    n_samples = len(trace.posterior.chain) * len(trace.posterior.draw)
    n_individuals = len(df)
    n_apoe_dummies = len(apoe_cols)
    
    # Extract posterior samples
    beta_edu_samples = trace.posterior['education'].values.flatten()
    beta_edu_age_samples = trace.posterior['education_age'].values.flatten()
    beta_edu_sex_samples = trace.posterior['education_sex'].values.flatten()
    beta_edu_apoe_samples = trace.posterior['education_APOE'].values.reshape(n_samples, n_apoe_dummies)
    
    # Calculate CATE for each individual and each posterior sample
    cate_samples = np.zeros((n_individuals, n_samples))
    
    for i in range(n_individuals):
        # Education × APOE interaction effect for this individual
        apoe_interaction = np.sum(beta_edu_apoe_samples * 
                                 df[apoe_cols].iloc[i].values.reshape(1, -1), axis=1)
        
        cate_logit = (beta_edu_samples + 
                     beta_edu_age_samples * df.iloc[i]['age_std'] +
                     beta_edu_sex_samples * df.iloc[i]['sex'] +
                     apoe_interaction)
        
        cate_samples[i, :] = cate_logit
    
    return cate_samples

# Calculate CATE estimates
cate_posterior = calculate_cate_posterior(trace, df)

# Convert cate_posterior (NumPy array) to a DataFrame
cate_posterior_df = pd.DataFrame(
    cate_posterior,
    columns=[f"sample_{i}" for i in range(cate_posterior.shape[1])]  # Name columns for posterior samples
)

# Add individual identifiers (e.g., IID or index)
cate_posterior_df['IID'] = df['IID'].values

# Save to Parquet file
cate_posterior_df.to_parquet('cate_posterior_samples.parquet', engine='fastparquet')

# # Summarize CATE estimates for each individual
# df['cate_mean'] = np.mean(cate_posterior, axis=1)
# df['cate_std'] = np.std(cate_posterior, axis=1)
# df['cate_lower'] = np.percentile(cate_posterior, 2.5, axis=1)
# df['cate_upper'] = np.percentile(cate_posterior, 97.5, axis=1)

# # Convert from log-odds to probability scale (approximate marginal effects)
# # Using average marginal effects approach
# baseline_p = df['AD_diagnosis'].mean()
# scale_factor = baseline_p * (1 - baseline_p)

# df['cate_prob_mean'] = df['cate_mean'] * scale_factor
# df['cate_prob_lower'] = df['cate_lower'] * scale_factor  
# df['cate_prob_upper'] = df['cate_upper'] * scale_factor

# print("\nCRATE Summary Statistics:")
# print(f"Mean CATE (log-odds): {df['cate_mean'].mean():.4f}")
# print(f"SD of CATEs: {df['cate_mean'].std():.4f}")
# print(f"Range: [{df['cate_mean'].min():.4f}, {df['cate_mean'].max():.4f}]")

# print(f"\nMean CATE (probability scale): {df['cate_prob_mean'].mean():.4f}")
# print(f"Range: [{df['cate_prob_mean'].min():.4f}, {df['cate_prob_mean'].max():.4f}]")

# # Visualize heterogeneity by effect modifiers
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# # CATE by Age
# axes[0,0].scatter(df['age'], df['cate_mean'], alpha=0.6)
# axes[0,0].set_xlabel('Age')
# axes[0,0].set_ylabel('CATE (log-odds)')
# axes[0,0].set_title('CATE by Age')

# # CATE by Sex
# sex_labels = ['Female', 'Male']
# for sex in [0, 1]:
#     mask = df['sex'] == sex
#     axes[0,1].scatter(np.random.normal(sex, 0.1, sum(mask)), 
#                      df.loc[mask, 'cate_mean'], 
#                      alpha=0.6, label=sex_labels[sex])
# axes[0,1].set_xlabel('Sex')
# axes[0,1].set_ylabel('CATE (log-odds)')
# axes[0,1].set_title('CATE by Sex')
# axes[0,1].set_xticks([0, 1])
# axes[0,1].set_xticklabels(sex_labels)
# axes[0,1].legend()

# # CATE by APOE genotype
# apoe_genotypes = sorted(df['APOE'].unique())
# colors = plt.cm.Set3(np.linspace(0, 1, len(apoe_genotypes)))

# for idx, apoe_type in enumerate(apoe_genotypes):
#     mask = df['APOE'] == apoe_type
#     if sum(mask) > 0:  # Only plot if category has observations
#         axes[1,0].scatter(np.random.normal(idx, 0.1, sum(mask)), 
#                          df.loc[mask, 'cate_mean'], 
#                          alpha=0.6, label=apoe_type, color=colors[idx])

# axes[1,0].set_xlabel('APOE Genotype')
# axes[1,0].set_ylabel('CATE (log-odds)')
# axes[1,0].set_title('CATE by APOE Genotype')
# axes[1,0].set_xticks(range(len(apoe_genotypes)))
# axes[1,0].set_xticklabels(apoe_genotypes, rotation=45)
# axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# # Distribution of CATEs
# axes[1,1].hist(df['cate_mean'], bins=50, alpha=0.7, edgecolor='black')
# axes[1,1].set_xlabel('CATE (log-odds)')
# axes[1,1].set_ylabel('Frequency')
# axes[1,1].set_title('Distribution of Individual CATEs')

# plt.tight_layout()
# plt.show()

# # Example: Focus on high-risk individuals (E4E4 or E3E4, older adults)
# high_risk = df[(df['APOE'].isin(['E4E4', 'E3E4'])) & (df['age'] > 75)]
# print(f"\nCATEs for high-risk individuals (E4E4 or E3E4, age >75, n={len(high_risk)}):")
# if len(high_risk) > 0:
#     print(f"Mean CATE: {high_risk['cate_prob_mean'].mean():.4f}")
#     print(f"Range: [{high_risk['cate_prob_mean'].min():.4f}, {high_risk['cate_prob_mean'].max():.4f}]")
# else:
#     print("No individuals in this high-risk category")

# # Save results
# print(f"\nFirst 10 individual CATE estimates:")
# display_cols = ['education', 'age', 'sex', 'APOE'] + ['cate_mean', 'cate_lower', 'cate_upper']
# print('10 lowest CATE estimates:')
# print(df[display_cols].sort_values(by='cate_mean').head(10))

# print('10 highest CATE estimates:')
# print(df[display_cols].sort_values(by='cate_mean').tail(10))