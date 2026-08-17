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

def on_pick(event):
    ind = event.ind[0]
    col = np.flatnonzero(mask_subtidal)[ind]

    MSL = 1000 * MSL_anomalies[:,col]
    
    plt.figure()
    plt.plot(bathy_years, MSL, c='k')
    plt.xlabel('Time')
    plt.ylabel('$\Delta$MSL w.r.t. 2009 bathymetry [mm]')

    return

def spatial_map(x, y, z, fig=None, ax=None, vmin=None, vmax=None, cmap=None, alpha=None, clabel='', title='', savename=''):
    if vmin is None and vmax is None:
        bound = np.nanpercentile(np.abs(z), 95)
        vmin = -bound
        vmax = bound
        
    if cmap == None:
        cmap = 'bwr'
        
    if alpha == None:
        alpha = 1

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
    fig.canvas.callbacks.connect('pick_event', on_pick)
    if savename != '':
        fig.savefig('../../figures/{}'.format(savename), dpi=300, bbox_inches='tight', pad_inches=0.05)
    return fig, ax



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




mask_subtidal = outputs[0]["mask_subtidal"].copy()

for d in outputs[1:]:
    mask_subtidal &= d["mask_subtidal"]


bound = np.nanpercentile(abs(MSL_trends[mask_subtidal]), 95)
spatial_map(lons[mask_subtidal], lats[mask_subtidal], MSL_trends[mask_subtidal], vmin=-bound, vmax=bound, clabel='MSL trend [mm/yr]', savename='MSL_trend.png', cmap=cm.vik)


def closest_N_idx(lons, lats, lon, lat, n):
    d2 = (lons - lon)**2 + (lats - lat)**2
    return np.argpartition(d2, n)[:n]

N = 1
indices_knock = closest_N_idx(lons, lats, coordinates_knock[0], coordinates_knock[1], N)
indices_knock = indices_knock[mask_subtidal[indices_knock]]

mean_MSL = np.mean(MSL_anomalies[:, indices_knock], axis=1)

plt.figure()
plt.plot(bathy_years, mean_MSL)
plt.title('N={}'.format(N))