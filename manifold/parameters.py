"""
File containing every parameter in the Manifold Filtering code.

Parameters

purkinje_radius: expected radius of a purkinje cell in the 3D image
distance_col: name of the reconstruction distance column in the markers file
x_col: name of the x coordinates column in the markers file
y_col: name of the y coordinates column in the markers file
z_col: name of the z coordinates column in the markers file
gt_x_col: name of the x coordinates column in the GT markers file
gt_y_col: name of the y coordinates column in the GT markers file
gt_z_col: name of the z coordinates column in the GT markers file
name_col: name of the name column in the markers file, should be something like "MS_CENTER ??(??????)"
substack_col: name of the substack ID column in the markers file
jobs: number of seeds folder to create, in order to parallelize patches reconstruction jobs
robust_iter: number of robust iterations of Lowess, in order to spot outliers
weight_threshold: threshold used by Lowess to discard points with too little weight in order to speed up
"""

purkinje_radius = 12
distance_col = 'distance'
x_col = '##x'
y_col = 'y'
z_col = 'z'
gt_x_col = '#x'
gt_y_col = ' y'
gt_z_col = ' z'
name_col = 'name'
substack_col = 'substack'
jobs = 64
robust_iter = 5
weight_threshold = 1e-4
