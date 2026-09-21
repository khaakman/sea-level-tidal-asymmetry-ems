#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

fourier_folder = '../../data/DCSM/fourier_dynamic_meteo_dynamic_bathymetry'
meteo_years = np.arange(1996, 2023)
partitions = [27, 28, 29]

global MSL_anomalies


outputs = []
for i, meteo_yr in enumerate(meteo_years):
    actual_folder = fourier_folder + '/A04_meteo{}_bathy{}_fourier/'.format(meteo_yr, meteo_yr)
    reference_folder = fourier_folder + '/A04_meteo{}_bathy2009_fourier/'.format(meteo_yr)
    
    out = {k: [] for k in [
        "MSL_anomaly"]}
        
    for j, partition_nr in enumerate(partitions):
        fn_actual = actual_folder + 'DCSM-FM_100m_00{}_fou.nc'.format(partition_nr)
        data_actual = xr.open_dataset(fn_actual)
        
        fn_reference = reference_folder + 'DCSM-FM_100m_00{}_fou.nc'.format(partition_nr)
        data_reference = xr.open_dataset(fn_reference)
        
        MSL_anomaly = data_actual.mesh2d_fourier001_mean.values - data_reference.mesh2d_fourier001_mean.values
        out["MSL_anomaly"].append(MSL_anomaly)

    # concatenate everything
    for k in out:
        if len(out[k]) > 0:
            out[k] = np.concatenate(out[k])
    
    outputs.append(out)


MSL_anomalies = 1000 * np.vstack([outputs[i]["MSL_anomaly"] for i in range(len(meteo_years))])

np.save('../../processed_time_series/DCSM/MSL_anomalies.npy', MSL_anomalies)
