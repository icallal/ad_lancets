import pandas as pd

def row_contains_values(row, values_set):
    return any(item in values_set for item in row)

def pull_rows_with_values(df, values_set):
    df = df[df.apply(lambda row: row_contains_values(row, values_set), axis=1)]
    return df

def pull_columns_by_prefix(df, column_prefixes):

    # Identify column names that start with any of the search strings
    matching_columns = [col for col in df.columns
                        if any(col.startswith(s)
                                for s in column_prefixes)]

    eid_in_columns = False
    # Ensure the exclude_column is in the matching_columns list
    if 'eid' in matching_columns:
        matching_columns.remove('eid')
        eid_in_columns = True
        
    # Filter rows where 1+ value in the subset of columns is not NA (NaN or None)
    df = df[~df[matching_columns].isnull().all(axis=1)]
    
    if eid_in_columns:
        # Include the exclude_column in the result
        df = df.loc[:, ['eid'] + matching_columns]
    else:
        df = df.loc[:, matching_columns]

    return df

def pull_columns_by_suffix(df, column_suffixes):

    # Identify column names that start with any of the search strings
    matching_columns = [col for col in df.columns
                        if any(col.endswith(s)
                                for s in column_suffixes)]

    eid_in_columns = False
    # Ensure the exclude_column is in the matching_columns list
    if 'eid' in matching_columns:
        matching_columns.remove('eid')
        eid_in_columns = True

    # Filter rows where 1+ value in the subset of columns is not NA (NaN or None)
    df = df[~df[matching_columns].isnull().all(axis=1)]
    
    if eid_in_columns:
        # Include the exclude_column in the result
        df = df.loc[:, ['eid'] + matching_columns]
    else:
        df = df.loc[:, matching_columns]

    return df




