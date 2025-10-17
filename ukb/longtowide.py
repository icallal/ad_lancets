import pandas as pd
import numpy as np
from cates_utils import save_cates_3d_array

import pandas as pd
import numpy as np
from cates_utils import save_cates_3d_array


# Alternative SUPER optimized version using pivot_table
def parquet_to_3d_array(parquet_path, output_npz_path):
    """
    Ultra-fast version using pandas pivot operations.
    """
    df = pd.read_parquet(parquet_path, engine='fastparquet')
    
    print(f"Original data shape: {df.shape}")
    
    # Get dimensions
    unique_patients = sorted(df['subject_id'].unique())
    unique_imaging = sorted(df['imaging_metric'].unique()) 
    unique_cognitive = sorted(df['cognitive_test'].unique())
    
    n_patients = len(unique_patients)
    n_imaging = len(unique_imaging)
    n_cognitive = len(unique_cognitive)
    
    print(f"Dimensions: {n_patients} x {n_imaging} x {n_cognitive}")
    
    # Initialize 3D array
    array_3d = np.full((n_patients, n_imaging, n_cognitive), np.nan, dtype=np.float32)
    
    print("Processing by cognitive test (ultra-fast)...")
    
    # Process each cognitive test separately using pivot
    for c_idx, cog_test in enumerate(unique_cognitive):
        print(f"  Processing {cog_test} ({c_idx+1}/{n_cognitive})")
        
        # Filter for this cognitive test
        cog_df = df[df['cognitive_test'] == cog_test]
        
        if len(cog_df) == 0:
            continue
            
        # Pivot to get patients × imaging_metrics matrix
        pivot_df = cog_df.pivot_table(
            index='subject_id', 
            columns='imaging_metric', 
            values='cate_value',
            aggfunc='first'  # In case of duplicates
        )
        
        # Reindex to ensure consistent ordering
        pivot_df = pivot_df.reindex(index=unique_patients, columns=unique_imaging)
        
        # Fill the 3D array slice
        array_3d[:, :, c_idx] = pivot_df.values
    
    # Check fill percentage
    n_filled = np.sum(~np.isnan(array_3d))
    n_total = np.prod(array_3d.shape)
    fill_percentage = (n_filled / n_total) * 100
    
    print(f"Array filled: {n_filled:,} / {n_total:,} ({fill_percentage:.1f}%)")
    
    # Save as NPZ
    save_cates_3d_array(
        array_3d=array_3d,
        imaging_metrics=unique_imaging,
        cognitive_tests=unique_cognitive,
        n_patients=n_patients,
        filepath=output_npz_path
    )
    
    return array_3d, unique_imaging, unique_cognitive, unique_patients

# Use the ultra-fast version
array_3d, imaging_metrics, cognitive_tests, patient_ids = parquet_to_3d_array(
    parquet_path='double_ml/imaging/results/imaging_to_cog/cates/merged_results.parquet',
    output_npz_path='double_ml/imaging/results/imaging_to_cog/cates/cates_3d_array.npz'
)

print(f"\n3D Array shape: {array_3d.shape}")
print(f"Memory usage: ~{array_3d.nbytes / 1024**2:.1f} MB")

# Optional: Quick verification
print(f"\nQuick stats:")
print(f"  Min CATE: {np.nanmin(array_3d):.4f}")
print(f"  Max CATE: {np.nanmax(array_3d):.4f}")
print(f"  Mean CATE: {np.nanmean(array_3d):.4f}")
print(f"  Std CATE: {np.nanstd(array_3d):.4f}")

# def parquet_to_3d_array(parquet_path, output_npz_path):
#     """
#     Convert long-format parquet file to 3D NPZ array.
    
#     Structure: [patients, imaging_metrics, cognitive_tests]
#     """
#     # Read the merged results
#     df = pd.read_parquet(parquet_path, engine='fastparquet')
    
#     print(f"Original data shape: {df.shape}")
#     print(f"Columns: {df.columns.tolist()}")
    
#     # Get unique values for each dimension
#     unique_patients = sorted(df['subject_id'].unique())
#     unique_imaging = sorted(df['imaging_metric'].unique()) 
#     unique_cognitive = sorted(df['cognitive_test'].unique())
    
#     n_patients = len(unique_patients)
#     n_imaging = len(unique_imaging)
#     n_cognitive = len(unique_cognitive)
    
#     print(f"Dimensions:")
#     print(f"  Patients: {n_patients}")
#     print(f"  Imaging metrics: {n_imaging}")
#     print(f"  Cognitive tests: {n_cognitive}")
    
#     # Initialize 3D array with NaN
#     array_3d = np.full((n_patients, n_imaging, n_cognitive), np.nan, dtype=np.float32)
    
#     # Create lookup dictionaries for fast indexing
#     patient_idx = {pid: i for i, pid in enumerate(unique_patients)}
#     imaging_idx = {img: i for i, img in enumerate(unique_imaging)}
#     cognitive_idx = {cog: i for i, cog in enumerate(unique_cognitive)}
    
#     # Fill the 3D array
#     print("Filling 3D array...")
#     for _, row in df.iterrows():
#         p_idx = patient_idx[row['subject_id']]
#         i_idx = imaging_idx[row['imaging_metric']]
#         c_idx = cognitive_idx[row['cognitive_test']]
#         array_3d[p_idx, i_idx, c_idx] = row['cate_value']
    
#     # Check how much data we have
#     n_filled = np.sum(~np.isnan(array_3d))
#     n_total = np.prod(array_3d.shape)
#     fill_percentage = (n_filled / n_total) * 100
    
#     print(f"Array filled: {n_filled:,} / {n_total:,} ({fill_percentage:.1f}%)")
    
#     # Save as NPZ
#     save_cates_3d_array(
#         array_3d=array_3d,
#         imaging_metrics=unique_imaging,
#         cognitive_tests=unique_cognitive,
#         n_patients=n_patients,
#         filepath=output_npz_path
#     )
    
#     return array_3d, unique_imaging, unique_cognitive, unique_patients

# # Convert your parquet file
# array_3d, imaging_metrics, cognitive_tests, patient_ids = parquet_to_3d_array(
#     parquet_path='double_ml/imaging/results/imaging_to_cog/cates/merged_results.parquet',
#     output_npz_path='double_ml/imaging/results/imaging_to_cog/cates/cates_3d_array.npz'
# )

# print(f"\n3D Array shape: {array_3d.shape}")
# print(f"Memory usage: ~{array_3d.nbytes / 1024**2:.1f} MB")

# # Optional: Quick verification
# print(f"\nQuick stats:")
# print(f"  Min CATE: {np.nanmin(array_3d):.4f}")
# print(f"  Max CATE: {np.nanmax(array_3d):.4f}")
# print(f"  Mean CATE: {np.nanmean(array_3d):.4f}")
# print(f"  Std CATE: {np.nanstd(array_3d):.4f}")