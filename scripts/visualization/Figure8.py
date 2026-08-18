#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:09:36 2026

@author: kbhaakman
"""

import numpy as np
import matplotlib.pyplot as plt

years = np.arange(1996, 2023)
ts_friction_M0_M2_anomalies = np.load('../../processed_time_series/DCSM/MSL_contributions/friction_M0_M2_anomalies.npy', )
ts_friction_M2_M4_anomalies = np.load('../../processed_time_series/DCSM/MSL_contributions/friction_M2_M4_anomalies.npy', )
MSL_anomalies = np.load('../../processed_time_series/DCSM/MSL_anomalies.npy')
lons = np.load('../../data/DCSM/lons.npy')
lats = np.load('../../data/DCSM/lats.npy')
mask_subtidal = np.load('../../data/DCSM/mask_subtidal.npy')

def closest_N_idx(lons, lats, lon, lat, n):
    d2 = (lons - lon)**2 + (lats - lat)**2
    return np.argpartition(d2, n)[:n]

coordinates_knock = (7.030674, 53.327173)

N = 1
indices_knock = closest_N_idx(lons, lats, coordinates_knock[0], coordinates_knock[1], N)
indices_knock = indices_knock[mask_subtidal[indices_knock]]

mean_MSL_anomaly = np.mean(MSL_anomalies[:, indices_knock], axis=1)

friction_sum_anomalies = ts_friction_M0_M2_anomalies + ts_friction_M2_M4_anomalies 

plt.figure(figsize=(10, 4))
plt.plot(years, friction_sum_anomalies, label='Sum of M0+M2 and M2+M4 frictional contribution')
plt.plot(years, mean_MSL_anomaly, color='k', label='Modeled MSL anomaly')
plt.ylabel('MSL anomaly [mm]')
plt.xlabel('Bathymetry year')
plt.title('Mean sea level anomaly at Knock')
plt.legend()
plt.savefig('../../figures/Figure8.png', dpi=300, bbox_inches='tight', pad_inches=0.05)