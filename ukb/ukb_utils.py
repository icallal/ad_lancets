import pandas as pd
import os
from pathlib import Path

def _list_directories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
def id_loc(list_of_fieldids):
    main_directory = '/n/groups/patel/uk_biobank/'

    directories = _list_directories(main_directory)

    target_file = 'fields.ukb'

    found = []
    for d in directories:
        if '22881' in d:
            continue
        dir_files = os.listdir(f'{main_directory}/{d}')

        if target_file in dir_files:
            df = pd.read_csv(f'{main_directory}/{d}/{target_file}', header=None)
            # print(df)

            df = df[df.iloc[:,0].isin(list_of_fieldids)]
            if len(df) > 0:
                found.extend(list(df.iloc[:,0].values))
                print(d, df.iloc[:,0].values)
                print(len(df))
                print('\n')
    print('Not found:')
    print(set(list_of_fieldids).difference(set(found)))

def load_demographics(data_path):
    # import age, sex, education, site, assessment date
    df = pd.read_parquet(data_path +
                                   'demographics/demographics_df.parquet'
                                   )
    return df

def remove_participants_full_missing(df, columns_to_check=None):
    if columns_to_check is None:
        columns_to_check = [col for col in df.columns.tolist() if col != 'eid']
    
    df_sub = df.loc[:, columns_to_check]
    
    na_counts = df_sub.isna().sum(axis=1)

    keep = na_counts[na_counts < len(columns_to_check)]

    df_keep = df.iloc[keep, :]

    return df_keep

def group_assessment_center(df, data_instance, assessment_centers_lookup):
    # convert region
    assessment_centers = pd.DataFrame(df.loc[:, f'54-{data_instance}.0'])
    assessment_centers = assessment_centers.merge(assessment_centers_lookup, left_on=f'54-{data_instance}.0', right_on='coding')

    assessment_centers['region_label'] = ''
    assessment_centers.loc[assessment_centers.meaning.isin(['Barts', 'Hounslow', 'Croydon']), 'region_label'] = 'London'
    assessment_centers.loc[assessment_centers.meaning.isin(['Wrexham', 'Swansea', 'Cardiff']), 'region_label'] = 'Wales'
    assessment_centers.loc[assessment_centers.meaning.isin(['Cheadle (imaging)', 'Cheadle (revisit)', 'Stockport',
                                                            'Stockport (pilot)', 'Manchester', 'Liverpool','Bury']),
                                                             'region_label'] = 'North-West'
    assessment_centers.loc[assessment_centers.meaning.isin(['Newcastle', 'Newcastle (imaging)', 'Middlesborough']), 'region_label'] = 'North-East'
    assessment_centers.loc[assessment_centers.meaning.isin(['Leeds','Sheffield']), 'region_label'] = 'Yorkshire and Humber'
    assessment_centers.loc[assessment_centers.meaning.isin(['Stoke','Birmingham']), 'region_label'] = 'West Midlands'
    assessment_centers.loc[assessment_centers.meaning.isin(['Nottingham']), 'region_label'] = 'East Midlands'
    assessment_centers.loc[assessment_centers.meaning.isin(['Oxford', 'Reading', 'Reading (imaging)']), 'region_label'] = 'South-East'
    assessment_centers.loc[assessment_centers.meaning.isin(['Bristol', 'Bristol (imaging)']), 'region_label'] = 'South-West'
    assessment_centers.loc[assessment_centers.meaning.isin(['Glasgow', 'Edinburgh']), 'region_label'] = 'Scotland'

    region_indices = assessment_centers.groupby('region_label').groups

    return region_indices

def get_last_completed_education(df, instance):
    # education columns have array indices 0-5 for each instance
    # for each patient, take the max value of the Instance 0 columns
    # to represent max education completed
    educ_cols = df.columns[df.columns.str.startswith(f'6138-{instance}')]
    max_educ = df.loc[:, educ_cols].max(axis=1)
    df['max_educ_complete'] = max_educ
    return df

def binary_encode_column_membership_datacoding2171(df, field_id_columns, new_column_name):
    '''
    Binary encode whether a patient has a history of a disease that comes from
    a source other than only self-report

    ENSURE that the column uses Data coding 2171: https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=2171
    '''
    member_eid = []

    for fid in field_id_columns:
        fid_df = df.loc[:, ['eid', fid]]
        fid_df = fid_df[fid_df[fid] != 50]
        fid_df.dropna(inplace=True)
        member_eid.extend(list(fid_df.eid))

    member_eid = list(set(member_eid))
    df[new_column_name] = df.eid.isin(member_eid).astype(int)

    return df


def get_protein_lookup():
    """ Load the protein lookup table from the coding143.tsv file.
    This file contains the mapping of protein IDs to their meanings, which may include multiple parts separated by semicolons.
    Returns:
        pd.DataFrame: A DataFrame containing the protein codes and their meanings, with the meanings split into separate columns.
    """
    # Load the coding143.tsv file from the metadata directory
    # Assuming the file is located two directories up from the current file's directory
    # Adjust the path as necessary based on your project structure
    # Use Path to ensure compatibility across different operating systems
    # Note: This assumes the file is located at ../../metadata/coding143.tsv relative to this script's location

    # Get the directory of the current file
    # and construct the path to the coding143.tsv file    
    try:
        module_dir = Path(__file__).resolve().parent
    except NameError:
        module_dir = Path.cwd() 
    protein_code = pd.read_csv(f'{module_dir}/../../../../randy/rfb/metadata/coding143.tsv', sep='\t')

    # Split the column by semicolon and expand into separate columns
    split_columns = protein_code['meaning'].str.split(';', expand=True)

    # Rename the new columns (optional)
    split_columns.columns = [f'part_{i+1}' for i in range(split_columns.shape[1])]

    # Concatenate the new columns with the original DataFrame (optional)
    protein_code = pd.concat([protein_code, split_columns], axis=1)

    # Drop the original column if no longer needed
    protein_code = protein_code.drop('meaning', axis=1)
    
    return protein_code    