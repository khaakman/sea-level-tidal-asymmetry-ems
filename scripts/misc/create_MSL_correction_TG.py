#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import re
import hatyan
plt.style.use('ggplot')

station = 'delfzijl'

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
    
    MSL_actual = np.zeros(len(years))
    MSL_ref = np.zeros(len(years))
    
    for i, fn in enumerate(years):
        print(years[i])
        fn_actual = "../../data/DCSM/hist_dynamic_meteo_dynamic_bathymetry/A04_meteo{}_bathy{}_DCSM-FM_100m_0000_his.nc".format(years[i], years[i])
        fn_reference = "../../data/DCSM/hist_dynamic_meteo_dynamic_bathymetry/A04_meteo{}_bathy2009_DCSM-FM_100m_0000_his.nc".format(years[i])
        data_actual,  station_names, t = load_data(fn_actual, years[i])
        data_reference, station_names, t = load_data(fn_reference, years[i])
 
        idx = np.where(station_names == DCSM_names[station])
        water_level_actual = np.ravel(data_actual.waterlevel.values[:,idx])
        amps_actual, phases_actual = harmonic_analysis_elevation(t, water_level_actual)
        
        water_level_ref = np.ravel(data_reference.waterlevel.values[:,idx])
        amps_ref, phases_ref = harmonic_analysis_elevation(t, water_level_ref)
        
        MSL_actual[i] = 100 * amps_actual[0]
        MSL_ref[i] = 100 * amps_ref[0]
        
    
    return lons, lats, MSL_actual, MSL_ref

def main():
    
    lons, lats, MSL_actual, MSL_ref = compute_single_meteo_year()
    
    plt.figure()
    plt.plot(years, MSL_ref, color='k', linestyle='solid', label='Modeled constant bathy') 
    plt.xlabel('Year')
    plt.title(station.capitalize())
    plt.legend()
    
    bathy_effect = MSL_actual - MSL_ref
    
    plt.figure()
    plt.plot(years, bathy_effect, color='k', linestyle='solid', label='Modeled bathymetry effect') 
    plt.xlabel('Year')
    plt.title(station.capitalize())
    plt.legend()
    #plt.savefig('../../figures/validation_tg.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    np.save('../../processed_time_series/DCSM/MSL_constant_bathymetry_{}.npy'.format(station), MSL_ref)
    np.save('../../processed_time_series/DCSM/MSL_dynamic_bathymetry_contribution_{}.png'.format(station), bathy_effect)
    
    
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



if __name__ == '__main__':
    main()
