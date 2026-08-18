#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:52:44 2026

@author: kbhaakman
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import re
import hatyan


plt.style.use('ggplot')

station = 'knock'



DCSM_names = {
  'borkumfischerbalje': 'BorkumFischerbalje',
  'borkumsudstrand': 'BORKUM_Sudstrand',
  'dukegat': 'DUKEGAT',
  'emshorn': 'EMSHORN',
  'emdenneueseeschleuse': 'EMDEN_Neue_Seeschl',
  'knock': 'KNOCK',
  'delfzijl': 'DELFZL',
  'pogum': 'POGUM'
}

### Constituent stuff ####
const_elev = hatyan.get_const_list_hatyan('year')
const_elev.remove('MNO5')

M2_idx = np.where(np.array(const_elev) == 'M2')[0][0]
M4_idx = np.where(np.array(const_elev) == 'M4')[0][0]

phase_lag_constits = ['M2', 'M4']
if phase_lag_constits == ['M2', 'M4']:
    phase_lag_string = '2M2-M4'
elif phase_lag_constits == ['M2', 'S2', 'MS4']:
    phase_lag_string = 'M2+S2-MS4'


def load_data(fn, yr):
    data = xr.open_dataset(fn).sel(time=slice("{}-12-31 23:00:00".format(yr-1), "{}-12-31 23:00:00".format(yr)))
    data = data.assign_coords(time=data.time + pd.Timedelta(hours=1)) #from UTC to UTC+1
    t = data.time.values

    
    names = np.array(data.station_name.values)
    names = names.astype(str)
    
    return data, names, t

def bathy_year(p):
    return int(re.search(r"bathy(\d{4})", p.name).group(1))

def compute_single_meteo_year():
    global years
    years = np.arange(1996, 2023)
    grid_indices = np.arange(1,1001)
    
    
    lons = np.zeros(len(grid_indices))
    lats = np.zeros(len(grid_indices))
    
    global phase_lags
    phase_lags = np.zeros((len(grid_indices), len(years)))
    phase_lags_TG = np.zeros(len(years))
    
    for i, fn in enumerate(years):
        print(years[i])
        fn = "../../data/DCSM/hist_dynamic_meteo_dynamic_bathymetry/A04_meteo{}_bathy{}_DCSM-FM_100m_0000_his.nc".format(years[i], years[i])
        data, station_names, t = load_data(fn, years[i])
        for k, grid_idx in enumerate(grid_indices):
            idx = np.where(station_names == 'ems{}'.format(grid_idx))
            water_level = np.ravel(data.waterlevel.values[:,idx])

            grid_lon = data.station_x_coordinate[idx]
            grid_lat = data.station_y_coordinate[idx]
        


            if i == 0 :
                lons[k] = grid_lon
                lats[k] = grid_lat
        idx = np.where(station_names == DCSM_names[station])
        water_level = np.ravel(data.waterlevel.values[:,idx])
        amps, phases = harmonic_analysis_elevation(t, water_level)
        phase_lags_TG[i] = compute_phase_lag(phases, phase_lag_constits, const_elev)
        
    
    return lons, lats,  phase_lags, phase_lags_TG

def main():
    
    lons, lats, _, phase_lags_DCSM = compute_single_meteo_year()

    np.save('../../processed_time_series/DCSM/{}_phase_lags_{}_dynamic_meteo.npy'.format(phase_lag_string, station), phase_lags_DCSM)
    

    # A = np.ones((len(years), 2))
    # A[:,1] = years
    # xhat_1996 = np.linalg.solve(A.T @ A, A.T @ phase_lags_TG_1996)
    # xhat_2022 = np.linalg.solve(A.T @ A, A.T @ phase_lags_TG_2022)
    
    plt.figure()
    plt.plot(years, phase_lags_DCSM, color='k', linestyle='solid', label='Modeled') 
    
    #plt.plot(years, A@xhat, 'r')
    plt.ylabel('{} phase lag [deg]'.format(phase_lag_string))
    plt.xlabel('Year')
    plt.title(station.capitalize())
    # plt.title('Borkum Südstrand')
    
    
    ## plot observations phase lag
    folder = '../../processed_time_series/tide_gauges/'
    phase_lags_obs = np.zeros(len(years))
    const_elev2 = const_elev.copy()
    const_elev2.remove('A0')
    for i in range(len(years)):
        try:
            phases = np.degrees(np.load(folder + '{}/phases_{}_{}.npy'.format(station, station, years[i])))
            phase_lags_obs[i] = compute_phase_lag(phases, phase_lag_constits, const_elev2)
        except FileNotFoundError:
            print(station, 'not found', years[i])
            phase_lags_obs[i] = np.nan
        
    
    plt.plot(years, phase_lags_obs, label='Observed')
    plt.legend()
    #plt.savefig('../../figures/validation_tg.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    plt.figure()
    plt.plot(years, phase_lags_DCSM, color='k', label='Numerical model')
    #plt.plot(years, -1*np.cos(np.radians(phase_lags_TG_2022)), color='b', label='DCSM meteo 2022')
    #plt.plot(years, A@xhat, 'r')
    plt.ylabel('$-\cos(${} phase lag) [-]'.format(phase_lag_string))
    plt.xlabel('Year')
    plt.title(station)
    
    
    ## plot observations phase lag
    folder = '../../processed_time_series/tide_gauges/'
    phase_lags_obs = np.zeros(len(years))
    const_elev2 = const_elev.copy()
    const_elev2.remove('A0')
    for i in range(len(years)):
        try:
            phases = np.degrees(np.load(folder + '{}/phases_{}_{}.npy'.format(station, station, years[i])))
            phase_lags_obs[i] = compute_phase_lag(phases, phase_lag_constits, const_elev2)
        except FileNotFoundError:
            phase_lags_obs[i] = np.nan
        
    
    plt.plot(years, phase_lags_obs, label='Observed')
    plt.legend()
    
    return

def harmonic_analysis_elevation(time, water_level):
    if np.sum(np.isfinite(water_level)) > 100:
        hatyan_df = pd.DataFrame({'values':water_level})
        hatyan_df.index = time
    
        hatyan_result = hatyan.analysis(hatyan_df, const_elev, nodalfactors=False, fu_alltimes=False)
        amps = hatyan_result.A.to_numpy() 
        phases = hatyan_result.phi_deg.to_numpy()
    else:
        amps = np.zeros(len(const_elev)) * np.nan
        phases = np.zeros(len(const_elev)) * np.nan
    
    
    return amps, phases

def compute_phase_lag(phases, phase_lag_constits, const_list):
    phase_idx = np.zeros(len(phase_lag_constits))
    
    for i, c in enumerate(phase_lag_constits):
        phase_idx[i] = np.where(np.array(const_list) == phase_lag_constits[i])[0][0]
    
    if len(phase_lag_constits) == 2:
        phase_lag = (2*phases[int(phase_idx[0])] - phases[int(phase_idx[1])]) % 360
    if len(phase_lag_constits) == 3:
        phase_lag = (phases[int(phase_idx[0])] + phases[int(phase_idx[1])] - phases[int(phase_idx[2])]) % 360
    
    return phase_lag


if __name__ == '__main__':
    main()
