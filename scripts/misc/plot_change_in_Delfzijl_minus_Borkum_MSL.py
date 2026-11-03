#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:44:54 2026

@author: kbhaakman
"""
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