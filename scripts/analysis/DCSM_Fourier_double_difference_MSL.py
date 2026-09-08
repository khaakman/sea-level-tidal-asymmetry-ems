#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:23:55 2026

@author: kbhaakman
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('ggplot')
import cmcrameri.cm as cm


fourier_folder = '../../data/DCSM/fourier_dynamic_meteo_dynamic_bathymetry'
meteo_years = np.arange(1996, 2023)
bathy_years = np.arange(1996, 2023)
partitions = [27, 28, 29]

coordinates_knock = (7.030674, 53.327173)

global mask_subtidal
global MSL_anomalies


outputs = []
for i, meteo_yr in enumerate(meteo_years):
    actual_folder = fourier_folder + '/A04_meteo{}_bathy{}_fourier/'.format(meteo_yr, meteo_yr)
    reference_folder = fourier_folder + '/A04_meteo{}_bathy2009_fourier/'.format(meteo_yr)
    
    out = {k: [] for k in [
        "lon","lat","MSL_anomaly","mask_subtidal"]}
        
    for j, partition_nr in enumerate(partitions):
        fn_actual = actual_folder + 'DCSM-FM_100m_00{}_fou.nc'.format(partition_nr)
        data_actual = xr.open_dataset(fn_actual)
        
        fn_reference = reference_folder + 'DCSM-FM_100m_00{}_fou.nc'.format(partition_nr)
        data_reference = xr.open_dataset(fn_reference)
        
        out["lon"].append(data_actual.mesh2d_face_x.values)
        out["lat"].append(data_actual.mesh2d_face_y.values)
        MSL_anomaly = data_actual.mesh2d_fourier001_mean.values - data_reference.mesh2d_fourier001_mean.values
        out["MSL_anomaly"].append(MSL_anomaly)
        mask_subtidal = (data_actual.mesh2d_fourier002_min_depth.values > 0.5) & (data_reference.mesh2d_fourier002_min_depth.values > 0.5)
        out["mask_subtidal"].append(mask_subtidal)
    # concatenate everything
    for k in out:
        if len(out[k]) > 0:
            out[k] = np.concatenate(out[k])
    
    outputs.append(out)


MSL_anomalies = 1000 * np.vstack([outputs[i]["MSL_anomaly"] for i in range(len(meteo_years))])
MSL_trends = np.polyfit(meteo_years, MSL_anomalies, deg=1)[0]
lons = outputs[0]['lon']
lats = outputs[0]['lat']

np.save('../../processed_time_series/DCSM/MSL_anomalies.npy', MSL_anomalies)


mask_subtidal = outputs[0]["mask_subtidal"].copy()

for d in outputs[1:]:
    mask_subtidal &= d["mask_subtidal"]


lons = lons[mask_subtidal]
lats = lats[mask_subtidal]
MSL_trends = MSL_trends[mask_subtidal]
MSL_anomalies = MSL_anomalies[:, mask_subtidal]

def closest_N_idx(lons, lats, lon, lat, n):
    d2 = (lons - lon)**2 + (lats - lat)**2
    return np.argpartition(d2, n)[:n]
#%%
N = 100
indices_knock = closest_N_idx(lons, lats, coordinates_knock[0], coordinates_knock[1], N)
#indices_knock = indices_knock[mask_subtidal[indices_knock]]

q25_trend = np.percentile(MSL_trends[indices_knock], 25)
median_trend = np.median(MSL_trends[indices_knock])
q75_trend = np.percentile(MSL_trends[indices_knock], 75)

plt.figure()
plt.hist(MSL_trends[indices_knock])
print('Median trend: {:.2f} mm/yr (IQR: {:.2f} - {:.2f})'.format(median_trend, q25_trend, q75_trend))
mean_MSL = np.mean(MSL_anomalies[:, indices_knock], axis=1)

plt.figure()
plt.plot(bathy_years, mean_MSL)
plt.title('N={}'.format(N))