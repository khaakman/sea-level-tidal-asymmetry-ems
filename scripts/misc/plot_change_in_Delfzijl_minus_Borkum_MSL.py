#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:44:54 2026

@author: kbhaakman
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

fn_borkum = '/Users/kbhaakman/Documents/PhD/sea-level-tidal-asymmetry-ems/processed_time_series/DCSM/MSL_dynamic_bathymetry_contribution_borkumfischerbalje.png.npy'
fn_delfzijl = '/Users/kbhaakman/Documents/PhD/sea-level-tidal-asymmetry-ems/processed_time_series/DCSM/MSL_dynamic_bathymetry_contribution_delfzijl.png.npy'

MSL_anomaly_borkum = np.load(fn_borkum)
MSL_anomaly_delfzijl = np.load(fn_delfzijl)

diff = MSL_anomaly_delfzijl - MSL_anomaly_borkum
diff -= diff[0]

years = np.arange(1996, 2023)

plt.figure()
plt.plot(years, diff)
plt.title('Change in Delfzijl minus Borkum MSL')
plt.ylabel('$\Delta$ MSL [cm]')

y = diff
y = diff

A = np.ones((len(years), 2))
A[:,1] = years
xhat = np.linalg.solve(A.T @ A, A.T @ y)
res_var = np.var(y - A@xhat)
Qxhxh = np.linalg.inv(A.T @ A) * res_var
std_trend = np.sqrt(Qxhxh[1,1])

plt.figure()
plt.plot(years, y - A@xhat)
plt.title('residual of fit')

print(diff)

print('Trend: {} pm {} mm/yr'.format(round(10*xhat[1],2), round(10*std_trend, 2)))

total_change = 26*xhat[1]
total_change_std = 26*std_trend
print('Total MSL change over 1996-2022: {} pm {} cm'.format(round(total_change,2), round(total_change_std,2)))
# %%
