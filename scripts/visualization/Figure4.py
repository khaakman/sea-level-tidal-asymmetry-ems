#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:14:44 2026

@author: kbhaakman
"""

import numpy as np
import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('ggplot')
import hatyan
const_list = hatyan.get_const_list_hatyan('year')
const_list.remove('MNO5')
const_list.remove('A0')
import sys
sys.path.append("../analysis") #to be able to load dictionary from paper_helper_functions.py
from paper_helper_functions import coordinates_ems

def compute_amp_phase_lag(constit, phase_lag_constits, years):
    amp_idx = np.where(np.array(const_list) == constit)[0][0]
    phase_idx = np.zeros(len(phase_lag_constits))
    
    amp_array = np.zeros((len(years), len(stations)))
    phase_array = np.zeros((len(years), len(stations)))
    phase_lag_array = np.zeros((len(years), len(stations)))
    
    for i, c in enumerate(phase_lag_constits):
        phase_idx[i] = np.where(np.array(const_list) == phase_lag_constits[i])[0][0]

        
    for i, station in enumerate(stations):
        for j, yr in enumerate(years):
            amps_fn = '../../processed_time_series/tide_gauges/{}/amps_{}_{}.npy'.format(station, station, yr)
            phases_fn = '../../processed_time_series/tide_gauges/{}/phases_{}_{}.npy'.format(station, station, yr)
            amps = np.load(amps_fn) * 100
            phases = np.degrees(np.load(phases_fn))
            amp_array[j, i] = amps[amp_idx]
            phase_array[j, i] = phases[amp_idx]

            if len(phase_lag_constits) == 2:
                phase_lag_array[j, i] = (2*phases[int(phase_idx[0])] - phases[int(phase_idx[1])]) % 360
            if len(phase_lag_constits) == 3:
                phase_lag_array[j, i] = (phases[int(phase_idx[0])] + phases[int(phase_idx[1])] - phases[int(phase_idx[2])]) % 360
    
    
    return amp_array, phase_array, phase_lag_array

def compute_trend(y):
    t = np.arange(len(y))
    t = t[np.isfinite(y)]
    y = y[np.isfinite(y)]
    
    A = np.ones((len(y), 2))
    A[:,1] = np.arange(len(y))
    xhat = np.linalg.solve(A.T @ A, A.T @ y)
    trend = xhat[1]

    return trend


# Load tide gauge data
stations = ['knock', 'delfzijl', 'emshorn', 'dukegat', 'borkumfischerbalje', 'borkumsudstrand',  'emdenneueseeschleuse', 'pogum',]
years = np.arange(2000, 2023)
lons_TG = np.zeros(len(stations))
lats_TG = np.zeros(len(stations))
constit = 'M2'
phase_lag_constits = ['M2', 'M4']

for i, name in enumerate(stations):
    lons_TG[i], lats_TG[i] = coordinates_ems[name]
    

amp_array, phase_array, phase_lag_array = compute_amp_phase_lag(constit, phase_lag_constits, years)

trends = np.zeros(len(stations))

for i in range(len(stations)):
    trends[i] = compute_trend(np.cos(np.radians(phase_lag_array[:,i])))



# Load DCSM post-processed data
M2_M4_elev_phase_2009_deg = np.load('../../processed_time_series/DCSM/harmonic_analysis/2M2_M4_elev_phase_2009.npy')
M2_M4_elev_phase_2009_cos = np.cos(np.radians(M2_M4_elev_phase_2009_deg))
ts_2M2_M4_elev_phase_anomalies = np.load('../../processed_time_series/DCSM/harmonic_analysis/2M2_M4_elev_phase_anomalies.npy')

lons = np.load('../../data/DCSM/lons.npy')
lats = np.load('../../data/DCSM/lats.npy')
subtidal_mask = np.load('../../data/DCSM/mask_subtidal.npy')
bathy_years = np.arange(1996, 2023)


ts_2M2_M4_elev_phase_wrt_2009_cos = np.cos(np.radians(ts_2M2_M4_elev_phase_anomalies + M2_M4_elev_phase_2009_deg))
M2_M4_elev_phase_trend = np.polyfit(bathy_years, ts_2M2_M4_elev_phase_wrt_2009_cos, deg=1)[0]


# Visualization
bound = np.nanpercentile(np.abs(M2_M4_elev_phase_trend[subtidal_mask]), 80)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(2, 1, 1, projection=ccrs.Mercator())
ax.coastlines()
ax.add_feature(cfeature.LAND)
ax.set_extent([6.5, 7.6, 53.2, 53.7], crs=ccrs.PlateCarree())

im = ax.scatter(lons_TG, lats_TG, c=trends, cmap=cm.vik, vmin=-bound, vmax=bound, s=100, transform=ccrs.PlateCarree())
bbox = ax.get_position()
cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
clb = plt.colorbar(im, cax=cax)
clb.ax.tick_params(labelsize=10)
clb.set_label('$\cos(2\phi^\eta_{M2} - \phi^\eta_{M4})$ trend [1/yr]', fontsize=10)
ax.text(0.02, 0.02, '(a)', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='bottom', ha='left')


ax = fig.add_subplot(2, 1, 2, projection=ccrs.Mercator())
ax.coastlines()
ax.add_feature(cfeature.LAND)
ax.set_extent([6.5, 7.6, 53.2, 53.7], crs=ccrs.PlateCarree())
im = ax.scatter(lons[subtidal_mask], lats[subtidal_mask], c=M2_M4_elev_phase_trend[subtidal_mask], vmin=-bound, vmax=bound, cmap=cm.vik, s=1, transform=ccrs.PlateCarree())
#ax.scatter(lons[~subtidal_mask], lats[~subtidal_mask], c=M2_M4_U_phase_trend[~subtidal_mask],  vmin=-bound, vmax=bound, cmap=cm.vik, s=1, transform=ccrs.PlateCarree(),alpha=0.03)
bbox = ax.get_position()
cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
clb = plt.colorbar(im, cax=cax)
clb.ax.tick_params(labelsize=10)
clb.set_label('$\cos(2\phi^\eta_{M2} - \phi^\eta_{M4})$ trend [1/yr]', fontsize=10)
ax.text(0.02, 0.02, '(b)', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='bottom', ha='left')


fig.savefig('../../figures/Figure4.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
