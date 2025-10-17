import numpy as np
import pandas as pd
# import pytest
# from unittest.mock import patch, MagicMock
import numpy as np
import os

def cates_to_3d_array(cates_dict):
    """
    Convert nested CATE dictionary to 3D array.
    
    Parameters:
    -----------
    cates_dict : dict
        Nested dictionary {imaging_metric: {cognitive_test: array, ...}, ...}
    
    Returns:
    --------
    tuple: (3d_array, imaging_metrics, cognitive_tests, n_patients)
        - 3d_array: shape (n_patients, n_imaging_metrics, n_cognitive_tests)
        - imaging_metrics: list of imaging metric names
        - cognitive_tests: list of cognitive test names  
        - n_patients: number of patients
    """
    if not cates_dict:
        return np.array([]), [], [], 0
    
    # Extract all unique imaging metrics and cognitive tests
    imaging_metrics = list(cates_dict.keys())
    all_cog_tests = set()
    
    for img_metric in imaging_metrics:
        all_cog_tests.update(cates_dict[img_metric].keys())
    
    cognitive_tests = sorted(list(all_cog_tests))
    
    # Determine number of patients from first available array
    n_patients = 0
    for img_metric in imaging_metrics:
        for cog_test in cognitive_tests:
            if cog_test in cates_dict[img_metric]:
                cate_array = cates_dict[img_metric][cog_test]
                if hasattr(cate_array, '__len__'):
                    n_patients = len(cate_array)
                    break
        if n_patients > 0:
            break
    
    if n_patients == 0:
        return np.array([]), imaging_metrics, cognitive_tests, 0
    
    # Initialize 3D array with NaN
    array_3d = np.full((n_patients, len(imaging_metrics), len(cognitive_tests)), np.nan)
    
    # Fill the 3D array
    for i, img_metric in enumerate(imaging_metrics):
        for j, cog_test in enumerate(cognitive_tests):
            if cog_test in cates_dict[img_metric]:
                cate_array = cates_dict[img_metric][cog_test]
                if hasattr(cate_array, '__len__') and len(cate_array) == n_patients:
                    array_3d[:, i, j] = cate_array.flatten() if hasattr(cate_array, 'flatten') else cate_array
    
    return array_3d, imaging_metrics, cognitive_tests, n_patients

# slicing functions

def get_patient_slice(array_3d, patient_idx, imaging_metrics, cognitive_tests):
    """Get all CATE values for a specific patient across all imaging metrics and cognitive tests."""
    return pd.DataFrame(array_3d[patient_idx, :, :], 
                       index=imaging_metrics, 
                       columns=cognitive_tests)

def get_imaging_slice(array_3d, imaging_idx, imaging_metrics, cognitive_tests):
    """Get CATE values for a specific imaging metric across all patients and cognitive tests."""
    return pd.DataFrame(array_3d[:, imaging_idx, :], 
                       columns=cognitive_tests)

def get_cognitive_slice(array_3d, cognitive_idx, imaging_metrics, cognitive_tests):
    """Get CATE values for a specific cognitive test across all patients and imaging metrics."""
    return pd.DataFrame(array_3d[:, :, cognitive_idx], 
                       columns=imaging_metrics)

# saving and loading

def save_cates_3d_array(array_3d, imaging_metrics, cognitive_tests, n_patients, filepath):
    """
    Save 3D CATE array and metadata efficiently using NPZ format.
    
    Parameters:
    -----------
    array_3d : np.ndarray
        3D array of shape (n_patients, n_imaging_metrics, n_cognitive_tests)
    imaging_metrics : list
        List of imaging metric names
    cognitive_tests : list
        List of cognitive test names
    n_patients : int
        Number of patients
    filepath : str
        Path to save the .npz file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    np.savez_compressed(
        filepath,
        cates_array=array_3d,
        imaging_metrics=np.array(imaging_metrics, dtype=object),
        cognitive_tests=np.array(cognitive_tests, dtype=object),
        n_patients=n_patients,
        # Store metadata as well
        array_shape=array_3d.shape,
        creation_timestamp=np.datetime64('now')
    )
    
    print(f"Saved 3D CATE array to {filepath}")
    print(f"Array shape: {array_3d.shape}")
    print(f"File size: {np.round(np.prod(array_3d.shape) * 4 / 1024**2, 2)} MB (estimated)")

def load_cates_3d_array(filepath):
    """
    Load 3D CATE array and metadata from NPZ file.
    
    Returns:
    --------
    tuple: (array_3d, imaging_metrics, cognitive_tests, n_patients)
    """
    data = np.load(filepath, allow_pickle=True)
    
    array_3d = data['cates_array']
    imaging_metrics = data['imaging_metrics'].tolist()
    cognitive_tests = data['cognitive_tests'].tolist()
    n_patients = int(data['n_patients'])
    
    print(f"Loaded 3D CATE array from {filepath}")
    print(f"Array shape: {array_3d.shape}")
    print(f"Imaging metrics: {len(imaging_metrics)}")
    print(f"Cognitive tests: {len(cognitive_tests)}")
    print(f"Patients: {n_patients}")
    
    return array_3d, imaging_metrics, cognitive_tests, n_patients

def covariates_to_add_cates(cog, img, covariates_df, cates_array, imaging_metrics, cognitive_tests):
    """
    Adds covariates to CATES, including patient IIDs. 
    """
    # Start with a copy to avoid modifying original
    df_subset = covariates_df[['eid', 'bmi', 'mdi', 'curr_age', 'e2/e2', 'e3/e3', 'e2/e3', 'e3/e4', 'e2/e4', 'e4/e4', 'education_years', cog, img]].copy()

    # Apply valid mask FIRST
    valid_mask = ~df_subset[cog].isna() & ~df_subset[img].isna()
    df_filtered = df_subset[valid_mask].reset_index(drop=True)  # Reset index after filtering!
    
    print(f"After filtering for {cog} and {img}: {len(df_filtered)} rows")
    
    # Get indices for 3D array
    imaging_idx = imaging_metrics.index(img)
    cogtest_idx = cognitive_tests.index(cog)
    
    # Extract CATE values - only for valid samples
    cates_array_sub = cates_array[:len(df_filtered), imaging_idx, cogtest_idx]
    
    print(f"CATE array subset length: {len(cates_array_sub)}")
    print(f"DataFrame length: {len(df_filtered)}")
    
    # Check lengths match
    if len(cates_array_sub) != len(df_filtered):
        print(f"WARNING: Length mismatch! CATE: {len(cates_array_sub)}, DataFrame: {len(df_filtered)}")
        # Take minimum length to avoid errors
        min_len = min(len(cates_array_sub), len(df_filtered))
        cates_array_sub = cates_array_sub[:min_len]
        df_filtered = df_filtered.iloc[:min_len]
        print(f"Truncated both to length: {min_len}")
    
    # Add CATE values
    df_filtered['cates_value'] = cates_array_sub
    
    return df_filtered


# # Tests
# class TestCatesTo3DArray:
    
#     def test_empty_dict(self):
#         """Test with empty input dictionary."""
#         array_3d, img_metrics, cog_tests, n_patients = cates_to_3d_array({})
#         assert array_3d.size == 0
#         assert img_metrics == []
#         assert cog_tests == []
#         assert n_patients == 0
    
#     def test_single_imaging_single_cognitive(self):
#         """Test with single imaging metric and single cognitive test."""
#         cates = {
#             'img1': {
#                 'cog1': np.array([1.0, 2.0, 3.0])
#             }
#         }
        
#         array_3d, img_metrics, cog_tests, n_patients = cates_to_3d_array(cates)
        
#         assert array_3d.shape == (3, 1, 1)
#         assert img_metrics == ['img1']
#         assert cog_tests == ['cog1']
#         assert n_patients == 3
#         assert np.array_equal(array_3d[:, 0, 0], [1.0, 2.0, 3.0])
    
#     def test_multiple_imaging_multiple_cognitive(self):
#         """Test with multiple imaging metrics and cognitive tests."""
#         cates = {
#             'img1': {
#                 'cog1': np.array([1.0, 2.0]),
#                 'cog2': np.array([3.0, 4.0])
#             },
#             'img2': {
#                 'cog1': np.array([5.0, 6.0]),
#                 'cog2': np.array([7.0, 8.0])
#             }
#         }
        
#         array_3d, img_metrics, cog_tests, n_patients = cates_to_3d_array(cates)
        
#         assert array_3d.shape == (2, 2, 2)
#         assert len(img_metrics) == 2
#         assert len(cog_tests) == 2
#         assert n_patients == 2
#         assert not np.isnan(array_3d).any()
    
#     def test_missing_combinations(self):
#         """Test with missing imaging-cognitive combinations."""
#         cates = {
#             'img1': {
#                 'cog1': np.array([1.0, 2.0])
#             },
#             'img2': {
#                 'cog2': np.array([3.0, 4.0])
#             }
#         }
        
#         array_3d, img_metrics, cog_tests, n_patients = cates_to_3d_array(cates)
        
#         assert array_3d.shape == (2, 2, 2)
#         # Check that missing combinations are NaN
#         img1_idx = img_metrics.index('img1')
#         img2_idx = img_metrics.index('img2')
#         cog1_idx = cog_tests.index('cog1')
#         cog2_idx = cog_tests.index('cog2')
        
#         assert np.isnan(array_3d[:, img1_idx, cog2_idx]).all()
#         assert np.isnan(array_3d[:, img2_idx, cog1_idx]).all()
    
#     def test_inconsistent_array_lengths(self):
#         """Test handling of inconsistent array lengths."""
#         cates = {
#             'img1': {
#                 'cog1': np.array([1.0, 2.0, 3.0]),
#                 'cog2': np.array([4.0, 5.0])  # Different length
#             }
#         }
        
#         array_3d, img_metrics, cog_tests, n_patients = cates_to_3d_array(cates)
        
#         # Should use the first valid array length as reference
#         assert n_patients == 3
#         assert array_3d.shape == (3, 1, 2)
#         # The shorter array should result in NaN values
#         assert np.isnan(array_3d[:, 0, cog_tests.index('cog2')]).any()

# class TestSliceFunctions:
    
#     def setup_method(self):
#         """Set up test data."""
#         self.cates = {
#             'img1': {
#                 'cog1': np.array([1.0, 2.0]),
#                 'cog2': np.array([3.0, 4.0])
#             },
#             'img2': {
#                 'cog1': np.array([5.0, 6.0]),
#                 'cog2': np.array([7.0, 8.0])
#             }
#         }
#         self.array_3d, self.img_metrics, self.cog_tests, self.n_patients = cates_to_3d_array(self.cates)
    
#     def test_get_patient_slice(self):
#         """Test getting patient slice."""
#         patient_df = get_patient_slice(self.array_3d, 0, self.img_metrics, self.cog_tests)
        
#         assert isinstance(patient_df, pd.DataFrame)
#         assert patient_df.shape == (2, 2)
#         assert list(patient_df.index) == self.img_metrics
#         assert list(patient_df.columns) == self.cog_tests
    
#     def test_get_imaging_slice(self):
#         """Test getting imaging slice."""
#         imaging_df = get_imaging_slice(self.array_3d, 0, self.img_metrics, self.cog_tests)
        
#         assert isinstance(imaging_df, pd.DataFrame)
#         assert imaging_df.shape == (2, 2)  # 2 patients, 2 cognitive tests
#         assert list(imaging_df.columns) == self.cog_tests
    
#     def test_get_cognitive_slice(self):
#         """Test getting cognitive slice."""
#         cognitive_df = get_cognitive_slice(self.array_3d, 0, self.img_metrics, self.cog_tests)
        
#         assert isinstance(cognitive_df, pd.DataFrame)
#         assert cognitive_df.shape == (2, 2)  # 2 patients, 2 imaging metrics
#         assert list(cognitive_df.columns) == self.img_metrics

# class TestExistingFunctions:
    
#     @patch('imaging_to_cog.pd.read_parquet')
#     def test_cates_to_dataframe(self, mock_read_parquet):
#         """Test the existing cates_to_dataframe function."""
#         cates = {
#             'img1': {
#                 'cog1': np.array([1.0, 2.0])
#             }
#         }
        
#         df = cates_to_dataframe(cates)
        
#         assert isinstance(df, pd.DataFrame)
#         assert 'imaging_metric' in df.columns
#         assert 'cognitive_test' in df.columns
#         assert 'cate_value' in df.columns
#         assert 'subject_id' in df.columns
    
#     def test_flatten_imaging_cog_dict_with_dataframes(self):
#         """Test flatten function with DataFrame inputs."""
#         mock_df = pd.DataFrame({
#             'coef': [0.5],
#             'std err': [0.1],
#             't': [5.0],
#             'P>|t|': [0.01],
#             '2.5 %': [0.3],
#             '97.5 %': [0.7]
#         })
        
#         data = {
#             'img1': {
#                 'cog1': mock_df
#             }
#         }
        
#         result = flatten_imaging_cog_dict(data)
        
#         assert isinstance(result, pd.DataFrame)
#         assert 'Imaging Metric' in result.columns
#         assert 'Cognitive Test' in result.columns
#         assert len(result) == 1

# if __name__ == "__main__":
#     pytest.main([__file__])