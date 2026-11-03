#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fourier_folder = '../../data/DCSM/fourier_dynamic_meteo_dynamic_bathymetry'
meteo_years = np.arange(1996, 2023)
partitions = [27, 28, 29]

coordinates = {}
coordinates['borkumfischerbalje'] = (6.747778, 53.557500)
coordinates['delfzijl'] = (6.933056, 53.326389) 
coordinates['knock'] = (7.030674, 53.327173)


global MSLs

def spatial_map(x, y, z, fig=None, ax=None, vmin=None, vmax=None, cmap=None, 
                alpha=None, p_values=None, p_threshold=0.05, clabel='', title='', savename=''):
    if vmin is None and vmax is None:
        bound = np.nanpercentile(np.abs(z), 95)
        vmin = -bound
        vmax = bound
        
    if cmap == None:
        cmap = 'bwr'
        
    if alpha == None:
        alpha = 1

    mask = np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if p_values is not None:
        p_values = p_values[mask]
    
    make_clb = 0
    if fig == None and ax == None:
        fig = plt.figure(figsize=(11.69,8.27))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        make_clb = 1
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.set_extent([6.5, 7.6, 53.2, 53.7], crs=ccrs.PlateCarree())
    
    im = ax.scatter(x, y, c=z, cmap=cmap, vmax=vmax, vmin=vmin, alpha=alpha, s=1, transform=ccrs.PlateCarree(), picker=True)


    
    if make_clb == 1:
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
        clb = plt.colorbar(im, cax=cax)
        clb.ax.tick_params(labelsize=16)
        clb.set_label(clabel, fontsize=16)
    ax.set_title(title)
    if savename != '':
        fig.savefig('../../figures/{}'.format(savename), dpi=300, bbox_inches='tight', pad_inches=0.05)
    return fig, ax


outputs = []
for i, meteo_yr in enumerate(meteo_years):
    reference_folder = fourier_folder + '/A04_meteo{}_bathy2009_fourier/'.format(meteo_yr)
    
    out = {k: [] for k in ["MSL"]}
        
    for j, partition_nr in enumerate(partitions):
        
        fn_reference = reference_folder + 'DCSM-FM_100m_00{}_fou.nc'.format(partition_nr)
        data_reference = xr.open_dataset(fn_reference)

        
        MSL = data_reference.mesh2d_fourier001_mean.values
        out["MSL"].append(MSL)

    # concatenate everything
    for k in out:
        if len(out[k]) > 0:
            out[k] = np.concatenate(out[k])
    
    outputs.append(out)


MSLs = 100 * np.vstack([outputs[i]["MSL"] for i in range(len(meteo_years))]) #to cm

omega_nodal = 2*np.pi / 18.613
A = np.ones((len(meteo_years), 4))
A[:,1] = meteo_years
# A[:,2] = np.cos(omega_nodal * meteo_years)
# A[:,3] = np.sin(omega_nodal * meteo_years)
A[:,2] = np.cos(omega_nodal * (meteo_years - 1922.7) + np.pi)
coeffs, residuals, rank, s = np.linalg.lstsq(A, MSLs, rcond=None)
# nodal_amplitudes = np.sqrt(coeffs[2]**2 + coeffs[3]**2)
nodal_amplitudes = coeffs[2]

lons = np.load('../../data/DCSM/lons.npy')
lats = np.load('../../data/DCSM/lats.npy')
mask_subtidal = np.load('../../data/DCSM/mask_subtidal.npy')

vmax = np.nanpercentile(nodal_amplitudes[mask_subtidal], 95)

spatial_map(lons[mask_subtidal], lats[mask_subtidal], nodal_amplitudes[mask_subtidal], vmax=vmax, clabel='Nodal amplitude [cm]',  savename='empirical_nodal_amplitude.png', vmin=0, cmap='viridis')

def closest_N_idx(lons, lats, lon, lat, n):
    d2 = (lons - lon)**2 + (lats - lat)**2
    return np.argpartition(d2, n)[:n]

MSLs = MSLs[:, mask_subtidal]
for station in ['borkumfischerbalje', 'knock', 'delfzijl']:
    N = 1
    index_station = closest_N_idx(lons[mask_subtidal], lats[mask_subtidal], coordinates[station][0], coordinates[station][1], N)
    
    print(MSLs[:, index_station])
    plt.figure()
    plt.plot(meteo_years, MSLs[:, index_station])
    plt.ylabel('DCSM sea level [cm]')
    plt.title(station)
#np.save('../../processed_time_series/DCSM/MSL_anomalies.npy', MSLs)
