#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:21:27 2026

@author: kbhaakman
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('ggplot')
import cmcrameri.cm as cm
from matplotlib.patches import Polygon
from paper_helper_functions import *

fourier_folder = '../../data/DCSM/fourier_dynamic_meteo_dynamic_bathymetry'
meteo_years = np.arange(1996, 2023)
bathy_years = np.arange(1996, 2023)
partitions = [27, 28, 29]

polygon_end = [
    (7.0765, 53.290),
    (7.078, 53.290),
    (7.078, 53.340),
    (7.0765, 53.340)
]

polygon_entrance = [
    (6.57091, 53.54424),
    (6.67411, 53.58290),
    (6.6733, 53.58366),
    (6.5701, 53.5450)
]


global mask_subtidal
g = 9.81

def on_pick(event):
    ind = event.ind[0]
    col = np.flatnonzero(mask_subtidal)[ind]

    ts = 1000 * friction_M2_M4_anomalies[:,col]
    
    plt.figure()
    plt.plot(bathy_years, ts, c='k')
    plt.xlabel('Time')
    plt.ylabel('Change in M2-M4 friction w.r.t. 2009 bathymetry [mm]')

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
        "lon","lat","advection_M0_anomaly", "advection_M2_anomaly",
        "advection_M4_anomaly", "advection_S2_anomaly", 
        "MSL_anomaly", "mask_subtidal","volume", "depth_actual", "depth_ref",
        "M0_u_amp_actual", "M0_v_amp_actual", "M0_u_amp_ref", "M0_v_amp_ref",
        "M2_u_amp_actual", "M2_v_amp_actual", "M2_u_amp_ref", "M2_v_amp_ref",
        "M4_u_amp_actual", "M4_v_amp_actual", "M4_u_amp_ref", "M4_v_amp_ref",
        "S2_u_amp_actual", "S2_v_amp_actual", "S2_u_amp_ref", "S2_v_amp_ref",
        "M2_u_phs_actual", "M2_v_phs_actual", "M2_u_phs_ref", "M2_v_phs_ref",
        "M4_u_phs_actual", "M4_v_phs_actual", "M4_u_phs_ref", "M4_v_phs_ref",
        "S2_u_phs_actual", "S2_v_phs_actual", "S2_u_phs_ref", "S2_v_phs_ref",
        "advection_M0_2009", "advection_M2_2009", "advection_M4_2009", 
        "advection_S2_2009", "mask_end"]}
        
    for j, partition_nr in enumerate(partitions):
        fn_actual = actual_folder + 'DCSM-FM_100m_00{}_fou.nc'.format(partition_nr)
        data_actual = xr.open_dataset(fn_actual)
        
        fn_reference = reference_folder + 'DCSM-FM_100m_00{}_fou.nc'.format(partition_nr)
        data_ref = xr.open_dataset(fn_reference)
        
        MSL_anomaly = data_actual.mesh2d_fourier001_mean.values - data_ref.mesh2d_fourier001_mean.values
        out["MSL_anomaly"].append(MSL_anomaly)
        
        lons = data_actual.mesh2d_face_x.values
        lats = data_actual.mesh2d_face_y.values
        
        out["lon"].append(lons)
        out["lat"].append(lats)
        
        
        depth_actual = abs(data_actual.mesh2d_flowelem_bl.values)
        out["depth_actual"].append(depth_actual)
        area_actual = data_actual.mesh2d_flowelem_ba.values
        
        M0_u_amp_actual = data_actual.mesh2d_fourier004_mean.values
        M0_v_amp_actual = data_actual.mesh2d_fourier006_mean.values
        out['M0_u_amp_actual'].append(M0_u_amp_actual)
        out['M0_v_amp_actual'].append(M0_v_amp_actual)
        
        M2_u_amp_actual = data_actual.mesh2d_fourier015_amp.values
        M2_v_amp_actual = data_actual.mesh2d_fourier017_amp.values
        M2_u_phs_actual = data_actual.mesh2d_fourier015_phs.values
        M2_v_phs_actual = data_actual.mesh2d_fourier017_phs.values
        out['M2_u_amp_actual'].append(M2_u_amp_actual)
        out['M2_v_amp_actual'].append(M2_v_amp_actual)
        out['M2_u_phs_actual'].append(M2_u_phs_actual)
        out['M2_v_phs_actual'].append(M2_v_phs_actual)
        
        M4_u_amp_actual = data_actual.mesh2d_fourier022_amp.values
        M4_v_amp_actual = data_actual.mesh2d_fourier023_amp.values
        M4_u_phs_actual = data_actual.mesh2d_fourier022_phs.values
        M4_v_phs_actual = data_actual.mesh2d_fourier023_phs.values
        out['M4_u_amp_actual'].append(M4_u_amp_actual)
        out['M4_v_amp_actual'].append(M4_v_amp_actual)
        out['M4_u_phs_actual'].append(M4_u_phs_actual)
        out['M4_v_phs_actual'].append(M4_v_phs_actual)
        
        S2_u_amp_actual = data_actual.mesh2d_fourier019_amp.values
        S2_v_amp_actual = data_actual.mesh2d_fourier020_amp.values
        S2_u_phs_actual = data_actual.mesh2d_fourier019_phs.values
        S2_v_phs_actual = data_actual.mesh2d_fourier020_phs.values
        out['S2_u_amp_actual'].append(S2_u_amp_actual)
        out['S2_v_amp_actual'].append(S2_v_amp_actual)
        out['S2_u_phs_actual'].append(S2_u_phs_actual)
        out['S2_v_phs_actual'].append(S2_v_phs_actual)
        
        
        
        
        
        depth_ref = abs(data_ref.mesh2d_flowelem_bl.values)
        out["depth_ref"].append(depth_ref)
        area_ref = data_ref.mesh2d_flowelem_ba.values
        out['volume'].append(depth_ref * area_ref)
        
        M0_u_amp_ref = data_ref.mesh2d_fourier004_mean.values
        M0_v_amp_ref = data_ref.mesh2d_fourier006_mean.values
        out['M0_u_amp_ref'].append(M0_u_amp_ref)
        out['M0_v_amp_ref'].append(M0_v_amp_ref)
        
        M2_u_amp_ref = data_ref.mesh2d_fourier015_amp.values
        M2_v_amp_ref = data_ref.mesh2d_fourier017_amp.values
        M2_u_phs_ref = data_ref.mesh2d_fourier015_phs.values
        M2_v_phs_ref = data_ref.mesh2d_fourier017_phs.values
        out['M2_u_amp_ref'].append(M2_u_amp_ref)
        out['M2_v_amp_ref'].append(M2_v_amp_ref)
        out['M2_u_phs_ref'].append(M2_u_phs_ref)
        out['M2_v_phs_ref'].append(M2_v_phs_ref)
        
        M4_u_amp_ref = data_ref.mesh2d_fourier022_amp.values
        M4_v_amp_ref = data_ref.mesh2d_fourier023_amp.values
        M4_u_phs_ref = data_ref.mesh2d_fourier022_phs.values
        M4_v_phs_ref = data_ref.mesh2d_fourier023_phs.values
        out['M4_u_amp_ref'].append(M4_u_amp_ref)
        out['M4_v_amp_ref'].append(M4_v_amp_ref)
        out['M4_u_phs_ref'].append(M4_u_phs_ref)
        out['M4_v_phs_ref'].append(M4_v_phs_ref)
        
        S2_u_amp_ref = data_ref.mesh2d_fourier019_amp.values
        S2_v_amp_ref = data_ref.mesh2d_fourier020_amp.values
        S2_u_phs_ref = data_ref.mesh2d_fourier019_phs.values
        S2_v_phs_ref = data_ref.mesh2d_fourier020_phs.values
        out['S2_u_amp_ref'].append(S2_u_amp_ref)
        out['S2_v_amp_ref'].append(S2_v_amp_ref)
        out['S2_u_phs_ref'].append(S2_u_phs_ref)
        out['S2_v_phs_ref'].append(S2_v_phs_ref)
        
        
        # out["friction_M2_M4_anomaly"].append(friction_M2_M4_anomaly)
        # out["friction_M2_M2_anomaly"].append(friction_M2_M2_anomaly)
        # out["friction_M0_M2_anomaly"].append(friction_M0_M2_anomaly)
        # out["friction_M0_S2_anomaly"].append(friction_M0_S2_anomaly)
        mask_subtidal = (data_actual.mesh2d_fourier002_min_depth.values > 0.5) & (data_ref.mesh2d_fourier002_min_depth.values > 0.5)
        out["mask_subtidal"].append(mask_subtidal)
    # concatenate everything
    for k in out:
        if len(out[k]) > 0:
            out[k] = np.concatenate(out[k])
    
    grid_points = np.vstack((out['lon'], out['lat'])).T
    poly_entrance = Polygon(polygon_entrance, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
    poly_end = Polygon(polygon_end, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
    mask_end = poly_end.get_path().contains_points(grid_points)
    mask_entrance = poly_entrance.get_path().contains_points(grid_points)
    mask_end &= out['mask_subtidal']
    mask_entrance &= out['mask_subtidal']
    out['mask_end'] = mask_end
    
    W_entrance, dy_entrance = cross_section_length(out['lon'], out['lat'], mask_entrance)
    W_end, dy_end = cross_section_length(out['lon'], out['lat'], mask_end)
    
    normal_entrance = cross_section_normal(out['lon'][mask_entrance], out['lat'][mask_entrance])
    normal_end = cross_section_normal(out['lon'][mask_end], out['lat'][mask_end])
    
    # TODO: store amplitudes and phases in 'out'; compute advective fluxes actual and ref bathymetries; compute advective anomaly w.r.t. 2009
    A_U_M0_entrance_actual = out['M0_u_amp_actual'][mask_entrance] * normal_entrance[0] + out['M0_v_amp_actual'][mask_entrance] * normal_entrance[1]
    A_U_M2_entrance_actual, _ = project_amplitude_direction(out['M2_u_amp_actual'][mask_entrance], out['M2_v_amp_actual'][mask_entrance], out['M2_u_phs_actual'][mask_entrance], out['M2_v_phs_actual'][mask_entrance], *normal_entrance)
    A_U_M4_entrance_actual, _ = project_amplitude_direction(out['M4_u_amp_actual'][mask_entrance], out['M4_v_amp_actual'][mask_entrance], out['M4_u_phs_actual'][mask_entrance], out['M4_v_phs_actual'][mask_entrance], *normal_entrance)
    A_U_S2_entrance_actual, _ = project_amplitude_direction(out['S2_u_amp_actual'][mask_entrance], out['S2_v_amp_actual'][mask_entrance], out['S2_u_phs_actual'][mask_entrance], out['S2_v_phs_actual'][mask_entrance], *normal_entrance)
             
    A_U_M0_end_actual = out['M0_u_amp_actual'][mask_end] * normal_end[0] + out['M0_v_amp_actual'][mask_end] * normal_end[1]
    A_U_M2_end_actual, _ = project_amplitude_direction(out['M2_u_amp_actual'][mask_end], out['M2_v_amp_actual'][mask_end], out['M2_u_phs_actual'][mask_end], out['M2_v_phs_actual'][mask_end], *normal_end)
    A_U_M4_end_actual, _ = project_amplitude_direction(out['M4_u_amp_actual'][mask_end], out['M4_v_amp_actual'][mask_end], out['M4_u_phs_actual'][mask_end], out['M4_v_phs_actual'][mask_end], *normal_end)
    A_U_S2_end_actual, _ = project_amplitude_direction(out['S2_u_amp_actual'][mask_end], out['S2_v_amp_actual'][mask_end], out['S2_u_phs_actual'][mask_end], out['S2_v_phs_actual'][mask_end], *normal_end)
    
    A_U_M0_entrance_ref = out['M0_u_amp_ref'][mask_entrance] * normal_entrance[0] + out['M0_v_amp_ref'][mask_entrance] * normal_entrance[1]
    A_U_M2_entrance_ref, _ = project_amplitude_direction(out['M2_u_amp_ref'][mask_entrance], out['M2_v_amp_ref'][mask_entrance], out['M2_u_phs_ref'][mask_entrance], out['M2_v_phs_ref'][mask_entrance], *normal_entrance)
    A_U_M4_entrance_ref, _ = project_amplitude_direction(out['M4_u_amp_ref'][mask_entrance], out['M4_v_amp_ref'][mask_entrance], out['M4_u_phs_ref'][mask_entrance], out['M4_v_phs_ref'][mask_entrance], *normal_entrance)
    A_U_S2_entrance_ref, _ = project_amplitude_direction(out['S2_u_amp_ref'][mask_entrance], out['S2_v_amp_ref'][mask_entrance], out['S2_u_phs_ref'][mask_entrance], out['S2_v_phs_ref'][mask_entrance], *normal_entrance)
             
    A_U_M0_end_ref = out['M0_u_amp_ref'][mask_end] * normal_end[0] + out['M0_v_amp_ref'][mask_end] * normal_end[1]
    A_U_M2_end_ref, _ = project_amplitude_direction(out['M2_u_amp_ref'][mask_end], out['M2_v_amp_ref'][mask_end], out['M2_u_phs_ref'][mask_end], out['M2_v_phs_ref'][mask_end], *normal_end)
    A_U_M4_end_ref, _ = project_amplitude_direction(out['M4_u_amp_ref'][mask_end], out['M4_v_amp_ref'][mask_end], out['M4_u_phs_ref'][mask_end], out['M4_v_phs_ref'][mask_end], *normal_end)
    A_U_S2_end_ref, _ = project_amplitude_direction(out['S2_u_amp_ref'][mask_end], out['S2_v_amp_ref'][mask_end], out['S2_u_phs_ref'][mask_end], out['S2_v_phs_ref'][mask_end], *normal_end)

    H_entrance_actual = out['depth_actual'][mask_entrance]
    H_entrance_ref = out['depth_ref'][mask_entrance]
    H_end_actual = out['depth_actual'][mask_end]
    H_end_ref = out['depth_ref'][mask_end]
    
    advection_actual = {}
    advection_actual['M0_entrance'] = 1/(2*g) * dy_entrance * np.nansum(H_entrance_actual * A_U_M0_entrance_actual**2)
    advection_actual['M2_entrance'] = 1/(2*g) * dy_entrance * np.nansum(H_entrance_actual * 0.5 * A_U_M2_entrance_actual**2)
    advection_actual['M4_entrance'] = 1/(2*g) * dy_entrance * np.nansum(H_entrance_actual * 0.5 * A_U_M4_entrance_actual**2)
    advection_actual['S2_entrance'] = 1/(2*g) * dy_entrance * np.nansum(H_entrance_actual * 0.5 * A_U_S2_entrance_actual**2)
    advection_actual['M0_end'] = 1/(2*g) * dy_end * np.nansum(H_end_actual * A_U_M0_end_actual**2)
    advection_actual['M2_end'] = 1/(2*g) * dy_end * np.nansum(H_end_actual * 0.5 * A_U_M2_end_actual**2)
    advection_actual['M4_end'] = 1/(2*g) * dy_end * np.nansum(H_end_actual * 0.5 * A_U_M4_end_actual**2)
    advection_actual['S2_end'] = 1/(2*g) * dy_end * np.nansum(H_end_actual * 0.5 * A_U_S2_end_actual**2)
    
    advection_ref = {}
    advection_ref['M0_entrance'] = 1/(2*g) * dy_entrance * np.nansum(H_entrance_ref * A_U_M0_entrance_ref**2)
    advection_ref['M2_entrance'] = 1/(2*g) * dy_entrance * np.nansum(H_entrance_ref * 0.5 * A_U_M2_entrance_ref**2)
    advection_ref['M4_entrance'] = 1/(2*g) * dy_entrance * np.nansum(H_entrance_ref * 0.5 * A_U_M4_entrance_ref**2)
    advection_ref['S2_entrance'] = 1/(2*g) * dy_entrance * np.nansum(H_entrance_ref * 0.5 * A_U_S2_entrance_ref**2)
    advection_ref['M0_end'] = 1/(2*g) * dy_end * np.nansum(H_end_ref * A_U_M0_end_ref**2)
    advection_ref['M2_end'] = 1/(2*g) * dy_end * np.nansum(H_end_ref * 0.5 * A_U_M2_end_ref**2)
    advection_ref['M4_end'] = 1/(2*g) * dy_end * np.nansum(H_end_ref * 0.5 * A_U_M4_end_ref**2)
    advection_ref['S2_end'] = 1/(2*g) * dy_end * np.nansum(H_end_ref * 0.5 * A_U_S2_end_ref**2)

    advection_grad_M0_actual = advection_actual['M0_end'] - advection_actual['M0_entrance']
    advection_grad_M2_actual = advection_actual['M2_end'] - advection_actual['M2_entrance']
    advection_grad_M4_actual = advection_actual['M4_end'] - advection_actual['M4_entrance']
    advection_grad_S2_actual = advection_actual['S2_end'] - advection_actual['S2_entrance']
    
    advection_grad_M0_ref = advection_ref['M0_end'] - advection_ref['M0_entrance']
    advection_grad_M2_ref = advection_ref['M2_end'] - advection_ref['M2_entrance']
    advection_grad_M4_ref = advection_ref['M4_end'] - advection_ref['M4_entrance']
    advection_grad_S2_ref = advection_ref['S2_end'] - advection_ref['S2_entrance']
    
    out['advection_M0_anomaly'] = advection_grad_M0_actual - advection_grad_M0_ref
    out['advection_M2_anomaly'] = advection_grad_M2_actual - advection_grad_M2_ref
    out['advection_M4_anomaly'] = advection_grad_M4_actual - advection_grad_M4_ref
    out['advection_S2_anomaly'] = advection_grad_S2_actual - advection_grad_S2_ref
    
    if meteo_yr == 2009:
        out['advection_M0_2009'] = advection_grad_M0_actual
        out['advection_M2_2009'] = advection_grad_M2_actual
        out['advection_M4_2009'] = advection_grad_M4_actual
        out['advection_S2_2009'] = advection_grad_S2_actual

    outputs.append(out)

mean_depth = np.mean(np.array([output["depth_actual"] for output in outputs]), axis=0)
np.save('../../data/DCSM/mean_depth.npy', mean_depth)

mask_subtidal = outputs[0]["mask_subtidal"].copy()
mask_end = outputs[0]['mask_end'].copy()
for d in outputs[1:]:
    mask_subtidal &= d["mask_subtidal"]
    mask_end &= d['mask_end']

np.save('../../data/DCSM/mask_end.npy', mask_end)

lons = outputs[0]['lon']
lats = outputs[0]['lat']

polygon_enclosed = [
    (6.5701, 53.5450),
    (6.69, 53.37),
    (7.078, 53.290),
    (7.078, 53.340),
    (7.00, 53.50),
    (6.6733, 53.58366)
]


poly_enclosed = Polygon(polygon_enclosed, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
grid_points = np.vstack((lons, lats)).T
mask_enclosed = poly_enclosed.get_path().contains_points(grid_points)

mask = mask_enclosed & mask_subtidal

ts_advection_anomaly_M0 = np.array([outputs[i]['advection_M0_anomaly'] for i in range(len(meteo_years))])
ts_advection_anomaly_M2 = np.array([outputs[i]['advection_M2_anomaly'] for i in range(len(meteo_years))])
ts_advection_anomaly_M4 = np.array([outputs[i]['advection_M4_anomaly'] for i in range(len(meteo_years))])
ts_advection_anomaly_S2 = np.array([outputs[i]['advection_S2_anomaly'] for i in range(len(meteo_years))])
ts_advection_M0 = ts_advection_anomaly_M0 + outputs[13]['advection_M0_2009']
ts_advection_M2 = ts_advection_anomaly_M2 + outputs[13]['advection_M2_2009']
ts_advection_M4 = ts_advection_anomaly_M4 + outputs[13]['advection_M4_2009']
ts_advection_S2 = ts_advection_anomaly_S2 + outputs[13]['advection_S2_2009']


L = compute_estuary_length_between_cross_sections(6.5701, 53.5450, 6.6733, 53.58366, 7.078, 53.290, 7.078, 53.340)
enclosed_volume = np.nansum(outputs[0]['volume'][mask])

ts_advection_anomaly_M0 = -1000 * ts_advection_anomaly_M0 / enclosed_volume * L
ts_advection_anomaly_M2 = -1000 * ts_advection_anomaly_M2 / enclosed_volume * L
ts_advection_anomaly_M4 = -1000 * ts_advection_anomaly_M4 / enclosed_volume * L
ts_advection_anomaly_S2 = -1000 * ts_advection_anomaly_S2 / enclosed_volume * L
ts_advection_M0 = -1000 * ts_advection_M0 / enclosed_volume * L
ts_advection_M2 = -1000 * ts_advection_M2 / enclosed_volume * L
ts_advection_M4 = -1000 * ts_advection_M4 / enclosed_volume * L
ts_advection_S2 = -1000 * ts_advection_S2 / enclosed_volume * L

plt.figure()
plt.plot(bathy_years, ts_advection_M0, label='M0')
plt.plot(bathy_years, ts_advection_M2, label='M2')
plt.plot(bathy_years, ts_advection_M4, label='M4')
plt.plot(bathy_years, ts_advection_S2, label='S2')
plt.legend()
plt.xlabel('Bathymetry year')
plt.ylabel('MSL contribution upstream [mm]')
np.save('../../processed_time_series/DCSM/MSL_contributions/advection_M0.npy', ts_advection_M0)
np.save('../../processed_time_series/DCSM/MSL_contributions/advection_M2.npy', ts_advection_M2)
np.save('../../processed_time_series/DCSM/MSL_contributions/advection_M4.npy', ts_advection_M4)
np.save('../../processed_time_series/DCSM/MSL_contributions/advection_S2.npy', ts_advection_S2)


plt.figure(figsize=(10, 4))
plt.plot(bathy_years, ts_advection_anomaly_M0, label='M0')
plt.plot(bathy_years, ts_advection_anomaly_M2, label='M2')
plt.plot(bathy_years, ts_advection_anomaly_M4, label='M4')
plt.plot(bathy_years, ts_advection_anomaly_S2, label='S2')
plt.legend()
plt.xlabel('Bathymetry year')
plt.ylabel('MSL anomaly contribution upstream [mm]')
plt.title('Advection')
#plt.savefig('../figures/advection_contributions.png', dpi=300, bbox_inches='tight')

