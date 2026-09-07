#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:00:31 2026

@author: kbhaakman
"""

import matplotlib.pyplot as plt
import numpy as np
import hatyan
import sys
sys.path.append("../analysis/") #to be able to load function from paper_helper_functions.py
from paper_helper_functions import compute_phase_lag
plt.style.use('ggplot')
const_elev = hatyan.get_const_list_hatyan('year')
const_elev.remove('MNO5')
from scipy.stats import t

###### Settings ########
phase_lag_constits = ['M2', 'M4']
phase_lag_string = '2M2-M4'
station1 = 'borkumsudstrand'
station2 = 'knock'
station1_display_name = 'Borkum Südstrand'
station2_display_name = 'Knock'
########################


years = np.arange(1996, 2023)

## get observed phase lags:
folder = '../../processed_time_series/tide_gauges/'
const_elev2 = const_elev.copy()
const_elev2.remove('A0')

def retrieve_observed_phase_lag(station):
    phase_lags_obs = np.zeros(len(years))
    for i in range(len(years)):
        try:
            phases = np.degrees(np.load(folder + '{}/phases_{}_{}.npy'.format(station, station, years[i])))
            phase_lags_obs[i] = compute_phase_lag(phases, phase_lag_constits, const_elev2)
        except FileNotFoundError:
            print(station, 'not found', years[i])
            phase_lags_obs[i] = np.nan
    return phase_lags_obs

def compute_trend(x, y):
    x = x[np.isfinite(y)]
    y = y[np.isfinite(y)]
    A = np.ones((len(y), 2))
    A[:,1] = x
    xhat = np.linalg.solve(A.T @ A, A.T @ y)
    trend = xhat[1]
    residuals = y - A@xhat
    Qxhxh = np.var(residuals) * np.linalg.inv(A.T @ A)
    trend_sigma = np.sqrt(Qxhxh[1,1])
    
    df = len(y) - 2 #degrees of freedom
    t_stat = trend / trend_sigma
    # Two-sided p-value
    p_value = 2 * t.sf(np.abs(t_stat), df=df)
    
    return A, xhat, trend_sigma, p_value

observed_phase_lags_station1 = retrieve_observed_phase_lag(station1)
observed_phase_lags_station2 = retrieve_observed_phase_lag(station2)
DCSM_phase_lags_station1_dynamic_meteo = np.load('../../processed_time_series/DCSM/{}_phase_lags_{}_dynamic_meteo.npy'.format(phase_lag_string, station1))
DCSM_phase_lags_station2_dynamic_meteo = np.load('../../processed_time_series/DCSM/{}_phase_lags_{}_dynamic_meteo.npy'.format(phase_lag_string, station2))

A, xhat_tg1, trend_sigma_tg1, _ = compute_trend(years, observed_phase_lags_station1)
A, xhat_tg2, trend_sigma_tg2, _ = compute_trend(years, observed_phase_lags_station2)

A, xhat_DCSM1, trend_sigma_DCSM1, _ = compute_trend(years[4:], DCSM_phase_lags_station1_dynamic_meteo[4:])
A, xhat_DCSM2, trend_sigma_DCSM2, _ = compute_trend(years[4:], DCSM_phase_lags_station2_dynamic_meteo[4:])

# Dynamic meteo plot
fig, axes = plt.subplots(nrows=2, figsize=(10, 7))
ax = axes[0]
ax.plot(years, observed_phase_lags_station1, label='Observed')
ax.plot(years[np.isfinite(observed_phase_lags_station1)], A@xhat_tg1, linestyle='dashed', 
        label='Fit ({:.2f} $\pm$ {:.2f} deg/yr'.format(xhat_tg1[1], trend_sigma_tg1), color='r')
ax.plot(years, DCSM_phase_lags_station1_dynamic_meteo, label='Modeled', color='k')
ax.plot(years[np.isfinite(observed_phase_lags_station1)], A@xhat_DCSM1, linestyle='dashed', 
        label='Fit ({:.2f} $\pm$ {:.2f} deg/yr'.format(xhat_DCSM1[1], trend_sigma_DCSM1), color='k')
ax.legend()
ax.set_title(station1_display_name)
ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=14, fontweight='bold',
             ha="left", va="top")


ax = axes[1]
ax.plot(years, observed_phase_lags_station2)
ax.plot(years[np.isfinite(observed_phase_lags_station2)], A@xhat_tg2, linestyle='dashed', 
        label='Fit ({:.2f} $\pm$ {:.2f} deg/yr'.format(xhat_tg2[1], trend_sigma_tg2), color='r')
ax.plot(years, DCSM_phase_lags_station2_dynamic_meteo, color='k')
ax.plot(years[np.isfinite(observed_phase_lags_station2)], A@xhat_DCSM2, linestyle='dashed', 
        label='Fit ({:.2f} $\pm$ {:.2f} deg/yr'.format(xhat_DCSM2[1], trend_sigma_DCSM2), color='k')
ax.set_xlabel('Year')
ax.set_title(station2_display_name)
ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=14, fontweight='bold',
             ha="left", va="top")
ax.legend()
plt.subplots_adjust(hspace=0.3)
fig.supylabel("{} elevation phase lag [deg]".format(phase_lag_string), x=0.05)
fig.savefig('../../figures/Figure3.png', dpi=300, bbox_inches='tight', pad_inches=0.05)