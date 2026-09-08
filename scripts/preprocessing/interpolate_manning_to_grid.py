#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

partitions = [27, 28, 29]
fourier_folder = '../../data/DCSM/fourier_dynamic_meteo_dynamic_bathymetry/A04_meteo1996_bathy1996_fourier/'

fn_manning = '../../data/DCSM/frictioncorrection_dcsm-fm_100m_v2_A02b_work48.xyz'
data = pd.read_csv(fn_manning, delim_whitespace=True, header=None)

lon_manning = data.iloc[:,0].values
lat_manning = data.iloc[:,1].values
manning = 0.028 * data.iloc[:,2].values

vmin = 0.012 #np.min(manning)
vmax = 0.023 #np.max(manning)
plt.figure()
plt.xlim(6, 8)
plt.ylim(53, 54)
im = plt.scatter(lon_manning, lat_manning, c=manning, vmin=vmin, vmax=vmax)
plt.colorbar(im)
plt.title('Original values')

coords_manning = np.column_stack((lon_manning, lat_manning))
tree = cKDTree(coords_manning)
plt.figure()
plt.xlim(6, 8)
plt.ylim(53, 54)
for i, partition in enumerate(partitions):
    fn = fourier_folder + 'DCSM-FM_100m_00{}_fou.nc'.format(partition)
    data = xr.open_dataset(fn)
    target_lon = data.mesh2d_face_x.values
    target_lat = data.mesh2d_face_y.values
    
    coords_target = np.column_stack((target_lon, target_lat))

    # Query nearest neighbour
    dist, idx = tree.query(coords_target)

    # Assign values
    interp_manning = manning[idx] 
    
    im = plt.scatter(target_lon, target_lat, c=interp_manning, vmin=vmin, vmax=vmax)
    savename = '../../data/DCSM/interpolated_manning_values_partition_{}.npy'.format(partition)
    np.save(savename, interp_manning)
    
plt.colorbar(im)
plt.title('Interpolated values')