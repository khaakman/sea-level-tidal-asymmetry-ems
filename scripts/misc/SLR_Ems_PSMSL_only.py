#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:18:29 2025

@author: kbhaakman
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

years = np.arange(1996, 2023)
stations = ['borkumfischerbalje', 'delfzijl']
correct_DCSM_surge = True
correct_DCSM_bottom = False
first_observation_zero = True
nodal_treatment = False #'empirical_fixed_phase'


def main():
    MSLs = np.zeros((len(stations), len(years)))
    for i, station in enumerate(stations):
        
        t, y = load_PSMSL_data(station, years[0], years[-1]+1)
        
        MSLs[i, :] = y
        
        plt.figure()
        plt.plot(years, y)
        plt.xlabel('Year')
        plt.ylabel('MSL [cm]')
        
        A, xhat = compute_trend(t, y)
        res_var = np.var(y - A@xhat)
        Qxhxh = np.linalg.inv(A.T @ A) * res_var
        std_trend = np.sqrt(Qxhxh[1,1])
        if nodal_treatment == 'empirical_variable_phase':
            amp_nodal = np.sqrt(xhat[2]**2 + xhat[3]**2)
        elif nodal_treatment == 'empirical_fixed_phase':
            amp_nodal = xhat[2]
        else:
            amp_nodal = 0
        plt.plot(years, A@xhat, 'k')
        plt.title('PSMSL {} (trend: {} $\pm$ {} mm/yr, nodal amp: {} cm)'.format(station.capitalize(), round(xhat[1]*10,1), round(std_trend*10, 1), round(amp_nodal, 1)))
        
        total_change = 26*xhat[1]
        total_change_std = 26*std_trend
        print(station)
        print('Total MSL change over 1996-2022: {} pm {} cm'.format(round(total_change,1), round(total_change_std,1)))
    
    difference = MSLs[-1,:] - MSLs[0,:]
    A, xhat = compute_trend(t, difference)
    res_var = np.var(difference - A@xhat)
    Qxhxh = np.linalg.inv(A.T @ A) * res_var
    std_trend = np.sqrt(Qxhxh[1,1])
    print('PSMSL difference (trend: {} $\pm$ {} mm/yr, nodal amp: {} cm)'.format( round(xhat[1]*10,1), round(std_trend*10, 1), round(amp_nodal, 1)))
    total_change = 26*xhat[1]
    total_change_std = 26*std_trend
    print('Total PSMSL difference over 1996-2022: {} pm {} cm'.format(round(total_change,1), round(total_change_std,1)))
    
    plt.figure()
    plt.scatter(years, difference)
    plt.plot(years, A@xhat, 'r')
    plt.xlabel('Year')
    plt.ylabel('Sea level difference [cm]')
    plt.title('{} minus {}'.format(stations[-1].capitalize(), stations[0].capitalize()))
    return

def compute_trend(t, y):
    omega_nodal = 2*np.pi / 18.613
    
    if nodal_treatment == 'empirical_variable_phase':
        ncols = 4
    elif nodal_treatment == 'empirical_fixed_phase':
        ncols = 3
    else:
        ncols = 2
    A = np.ones((len(t), ncols))
    A[:,1] = t
    if nodal_treatment == 'empirical_variable_phase':
        A[:,2] = np.cos(omega_nodal*t)
        A[:,3] = np.sin(omega_nodal*t)
    elif nodal_treatment == 'empirical_fixed_phase':
        A[:,2] = np.cos(omega_nodal * (t - 1922.7) + np.pi)
    xhat = np.linalg.solve(A.T @ A, A.T @ y)
    return A, xhat

def load_PSMSL_data(station, t0, t1):
    fn = '/Users/kbhaakman/Documents/PhD/data/PSMSL/{}_yearly.rlrdata'.format(station.capitalize())
    data = np.loadtxt(fn, delimiter=';', usecols=[0,1])
    t = data[:,0]
    y = data[:,1] / 10 #convert mm to cm
    
    
    # Only keep data between t0 and t1
    idx = np.where(t >= t0)
    y = y[idx]
    t = t[idx]
    
    idx = np.where(t < t1)
    y = y[idx]
    t = t[idx]
    
    plt.figure()
    plt.plot(t, y, label='Uncorrected')
    plt.ylabel('MSL [cm]')
    plt.xlabel('Year')
    plt.title(station)

    
    
    if correct_DCSM_surge:
        surge = np.load('/Users/kbhaakman/Documents/PhD/data/DCSM/yearly_surge_Ems_1996_2022/yearly_surge_{}.npy'.format(station))
        surge -= surge[-1]
        y -= np.ravel(surge)
        plt.plot(t, y, label='Minus surge')
        
    if correct_DCSM_bottom:
        bottom_effect = np.load('MSL_DCSM_{}_meteo_2022.npy'.format(station)) * 100
        
        
        bottom_effect -= bottom_effect[-1]
        y -= bottom_effect
        plt.plot(t, y, label='Minus surge and bathymetry')
    plt.legend()
    
    if first_observation_zero:
        y -= y[0]
    return t, y


if __name__ == '__main__':
    main()
# %%
