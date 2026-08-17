#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:50:37 2026

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
from paper_helper_functions import compute_estuary_length_between_cross_sections, project_residual_current_centerline, project_amplitudes_centerline

fourier_folder = '../../data/DCSM/fourier_dynamic_meteo_dynamic_bathymetry'
meteo_years = np.arange(1996, 2023)
bathy_years = np.arange(1996, 2023)
partitions = [27, 28, 29]


global mask_subtidal
global friction_M2_M4_anomalies
global friction_M0_M2_anomalies

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
        fig.savefig('/Users/kbhaakman/Desktop/{}'.format(savename), dpi=300, bbox_inches='tight', pad_inches=0.05)
    return fig, ax



outputs = []
for i, meteo_yr in enumerate(meteo_years):
    actual_folder = fourier_folder + '/A04_meteo{}_bathy{}_fourier/'.format(meteo_yr, meteo_yr)
    reference_folder = fourier_folder + '/A04_meteo{}_bathy2009_fourier/'.format(meteo_yr)
    
    out = {k: [] for k in [
        "lon","lat", "2M2_M4_U_phase_anomaly","2M2_M4_elev_phase_anomaly",
        "M2_U_phase_2009", "M4_U_phase_2009", "M2_elev_phase_2009", "M4_elev_phase_2009",
        "friction_M0_M2_anomaly", "friction_M2_M2_anomaly",
        "friction_M2_M4_anomaly", "friction_M0_S2_anomaly", 
        "MSL_anomaly", "mask_subtidal","volume",
        "friction_M0_M2_2009", "friction_M2_M2_2009", "friction_M2_M4_2009",
        "friction_M0_S2_2009"]}
        
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
        
        
        fn_manning = '../../data/DCSM/interpolated_manning_values_partition_{}.npy'.format(partition_nr)
        mannings = np.load(fn_manning)
        
        
        depth_actual = abs(data_actual.mesh2d_flowelem_bl.values)
        area_actual = data_actual.mesh2d_flowelem_ba.values
        out['volume'].append(depth_actual * area_actual)
        
        M0_u_amp_actual = data_actual.mesh2d_fourier004_mean.values
        M0_v_amp_actual = data_actual.mesh2d_fourier006_mean.values
        M0_U_amp_actual = project_residual_current_centerline(lons, lats, M0_u_amp_actual, M0_v_amp_actual)
        
        M2_elev_amp_actual = data_actual.mesh2d_fourier013_amp.values
        M2_elev_phs_actual = data_actual.mesh2d_fourier013_phs.values
        
        M2_u_amp_actual = data_actual.mesh2d_fourier015_amp.values
        M2_v_amp_actual = data_actual.mesh2d_fourier017_amp.values
        M2_u_phs_actual = data_actual.mesh2d_fourier015_phs.values
        M2_v_phs_actual = data_actual.mesh2d_fourier017_phs.values
        M2_U_amp_actual, M2_U_phs_actual, _, _ = project_amplitudes_centerline(lons, lats, M2_u_amp_actual, M2_v_amp_actual, M2_u_phs_actual, M2_v_phs_actual)
        
        M4_elev_phs_actual = data_actual.mesh2d_fourier021_phs.values
        
        M4_u_amp_actual = data_actual.mesh2d_fourier022_amp.values
        M4_v_amp_actual = data_actual.mesh2d_fourier023_amp.values
        M4_u_phs_actual = data_actual.mesh2d_fourier022_phs.values
        M4_v_phs_actual = data_actual.mesh2d_fourier023_phs.values
        M4_U_amp_actual, M4_U_phs_actual, _, _ = project_amplitudes_centerline(lons, lats, M4_u_amp_actual, M4_v_amp_actual, M4_u_phs_actual, M4_v_phs_actual)
        
        S2_u_amp_actual = data_actual.mesh2d_fourier019_amp.values
        S2_v_amp_actual = data_actual.mesh2d_fourier020_amp.values
        S2_u_phs_actual = data_actual.mesh2d_fourier019_phs.values
        S2_v_phs_actual = data_actual.mesh2d_fourier020_phs.values
        S2_U_amp_actual, S2_U_phs_actual, _, _ = project_amplitudes_centerline(lons, lats, S2_u_amp_actual, S2_v_amp_actual, S2_u_phs_actual, S2_v_phs_actual)
        
        
        
        depth_ref = abs(data_ref.mesh2d_flowelem_bl.values)
        area_ref = data_ref.mesh2d_flowelem_ba.values
        
        
        M0_u_amp_ref = data_ref.mesh2d_fourier004_mean.values
        M0_v_amp_ref = data_ref.mesh2d_fourier006_mean.values
        M0_U_amp_ref = project_residual_current_centerline(lons, lats, M0_u_amp_ref, M0_v_amp_ref)
        
        M2_elev_amp_ref = data_ref.mesh2d_fourier013_amp.values
        M2_elev_phs_ref = data_ref.mesh2d_fourier013_phs.values
        
        M2_u_amp_ref = data_ref.mesh2d_fourier015_amp.values
        M2_v_amp_ref = data_ref.mesh2d_fourier017_amp.values
        M2_u_phs_ref = data_ref.mesh2d_fourier015_phs.values
        M2_v_phs_ref = data_ref.mesh2d_fourier017_phs.values
        M2_U_amp_ref, M2_U_phs_ref, _, _ = project_amplitudes_centerline(lons, lats, M2_u_amp_ref, M2_v_amp_ref, M2_u_phs_ref, M2_v_phs_ref)
        
        M4_elev_phs_ref = data_ref.mesh2d_fourier021_phs.values
        
        M4_u_amp_ref = data_ref.mesh2d_fourier022_amp.values
        M4_v_amp_ref = data_ref.mesh2d_fourier023_amp.values
        M4_u_phs_ref = data_ref.mesh2d_fourier022_phs.values
        M4_v_phs_ref = data_ref.mesh2d_fourier023_phs.values
        M4_U_amp_ref, M4_U_phs_ref, _, _ = project_amplitudes_centerline(lons, lats, M4_u_amp_ref, M4_v_amp_ref, M4_u_phs_ref, M4_v_phs_ref)
        
        S2_u_amp_ref = data_ref.mesh2d_fourier019_amp.values
        S2_v_amp_ref = data_ref.mesh2d_fourier020_amp.values
        S2_u_phs_ref = data_ref.mesh2d_fourier019_phs.values
        S2_v_phs_ref = data_ref.mesh2d_fourier020_phs.values
        S2_U_amp_ref, S2_U_phs_ref, _, _ = project_amplitudes_centerline(lons, lats, S2_u_amp_ref, S2_v_amp_ref, S2_u_phs_ref, S2_v_phs_ref)
        
        M2_M4_U_phase_actual = (2*M2_U_phs_actual - M4_U_phs_actual) % 360
        M2_M4_U_phase_ref = (2*M2_U_phs_ref - M4_U_phs_ref) % 360
        M2_M4_U_phase_anomaly = M2_M4_U_phase_actual - M2_M4_U_phase_ref
        out['2M2_M4_U_phase_anomaly'].append(M2_M4_U_phase_anomaly)
        
        M2_M4_elev_phase_actual = (2*M2_elev_phs_actual - M4_elev_phs_actual) % 360
        M2_M4_elev_phase_ref = (2*M2_elev_phs_ref - M4_elev_phs_ref) % 360
        M2_M4_elev_phase_anomaly = M2_M4_elev_phase_actual - M2_M4_elev_phase_ref
        out['2M2_M4_elev_phase_anomaly'].append(M2_M4_elev_phase_anomaly)

        friction_M2_M2_actual = -depth_actual * area_actual * 16 / (9*np.pi) * mannings**2 / depth_actual**(7/3) * M2_elev_amp_actual * M2_U_amp_actual**2 * np.cos(np.radians(M2_elev_phs_actual - M2_U_phs_actual))
        friction_M2_M2_ref = -depth_ref * area_ref * 16 / (9*np.pi) * mannings**2 / depth_ref**(7/3) * M2_elev_amp_ref * M2_U_amp_ref**2 * np.cos(np.radians(M2_elev_phs_ref - M2_U_phs_ref))
        friction_M2_M2_anomaly = friction_M2_M2_actual - friction_M2_M2_ref
        
        friction_M2_M4_actual = depth_actual * 4/(3*np.pi) * area_actual * mannings**2 / depth_actual**(4/3)  * M2_U_amp_actual * M4_U_amp_actual * np.cos(np.radians(M4_U_phs_actual - 2*M2_U_phs_actual))
        friction_M2_M4_ref = depth_ref * 4/(3*np.pi) * area_ref * mannings**2 / depth_ref**(4/3)  * M2_U_amp_ref * M4_U_amp_ref * np.cos(np.radians(M4_U_phs_ref - 2*M2_U_phs_ref))
        friction_M2_M4_anomaly = friction_M2_M4_actual - friction_M2_M4_ref
        
        friction_M0_M2_actual = depth_actual * 4/(np.pi) * area_actual * mannings**2 / depth_actual**(4/3) * M0_U_amp_actual * M2_U_amp_actual
        friction_M0_M2_ref = depth_ref * 4/(np.pi) * area_ref * mannings**2 / depth_ref**(4/3) * M0_U_amp_ref * M2_U_amp_ref
        friction_M0_M2_anomaly = friction_M0_M2_actual - friction_M0_M2_ref
        
        friction_M0_S2_actual = depth_actual * 4/(np.pi) * area_actual * mannings**2 / depth_actual**(4/3) * M0_U_amp_actual * S2_U_amp_actual
        friction_M0_S2_ref = depth_ref * 4/(np.pi) * area_ref * mannings**2 / depth_ref**(4/3) * M0_U_amp_ref * S2_U_amp_ref
        friction_M0_S2_anomaly = friction_M0_S2_actual - friction_M0_S2_ref
        
        out["friction_M2_M4_anomaly"].append(friction_M2_M4_anomaly)
        out["friction_M2_M2_anomaly"].append(friction_M2_M2_anomaly)
        out["friction_M0_M2_anomaly"].append(friction_M0_M2_anomaly)
        out["friction_M0_S2_anomaly"].append(friction_M0_S2_anomaly)
        mask_subtidal = (data_actual.mesh2d_fourier002_min_depth.values > 0.5) & (data_ref.mesh2d_fourier002_min_depth.values > 0.5)
        out["mask_subtidal"].append(mask_subtidal)
        
        if meteo_yr == 2009:
            out['friction_M0_M2_2009'].append(friction_M0_M2_actual)
            out['friction_M2_M4_2009'].append(friction_M2_M4_actual)
            out['friction_M2_M2_2009'].append(friction_M2_M2_actual)
            out['friction_M0_S2_2009'].append(friction_M0_S2_actual)
            
            out['M2_U_phase_2009'].append(M2_U_phs_actual)
            out['M4_U_phase_2009'].append(M4_U_phs_actual)
            
            out['M2_elev_phase_2009'].append(M2_elev_phs_actual)
            out['M4_elev_phase_2009'].append(M4_elev_phs_actual)
            
        data_actual.close()
        data_ref.close()
    # concatenate everything
    for k in out:
        if len(out[k]) > 0:
            out[k] = np.concatenate(out[k])
    
    outputs.append(out)

M2_U_phase_2009 = outputs[13]['M2_U_phase_2009']
M4_U_phase_2009 = outputs[13]['M4_U_phase_2009']
M2_M4_U_phase_2009 = (2*M2_U_phase_2009 - M4_U_phase_2009) % 360
ts_2M2_M4_U_phase_anomalies = np.vstack([outputs[i]["2M2_M4_U_phase_anomaly"] for i in range(len(meteo_years))])

M2_elev_phase_2009 = outputs[13]['M2_elev_phase_2009']
M4_elev_phase_2009 = outputs[13]['M4_elev_phase_2009']
M2_M4_elev_phase_2009 = (2*M2_elev_phase_2009 - M4_elev_phase_2009) % 360
ts_2M2_M4_elev_phase_anomalies = np.vstack([outputs[i]["2M2_M4_elev_phase_anomaly"] for i in range(len(meteo_years))])


np.save('../../processed_time_series/DCSM/harmonic_analysis/2M2_M4_U_phase_2009.npy', M2_M4_U_phase_2009)
np.save('../../processed_time_series/DCSM/harmonic_analysis/2M2_M4_U_phase_anomalies.npy', ts_2M2_M4_U_phase_anomalies)

np.save('../../processed_time_series/DCSM/harmonic_analysis/2M2_M4_elev_phase_2009.npy', M2_M4_elev_phase_2009)
np.save('../../processed_time_series/DCSM/harmonic_analysis/2M2_M4_elev_phase_anomalies.npy', ts_2M2_M4_elev_phase_anomalies)

lons = outputs[0]['lon']
lats = outputs[0]['lat']

np.save('../../data/DCSM/lons.npy', lons)
np.save('../../data/DCSM/lats.npy', lats)

MSL_anomalies = 1000 * np.vstack([outputs[i]["MSL_anomaly"] for i in range(len(meteo_years))])


friction_M2_M4_anomalies = np.vstack([outputs[i]["friction_M2_M4_anomaly"] for i in range(len(meteo_years))])
#friction_M2_M4_trends = np.polyfit(meteo_years, friction_M2_M4_anomalies, deg=1)[0]

friction_M0_M2_anomalies = np.vstack([outputs[i]["friction_M0_M2_anomaly"] for i in range(len(meteo_years))])
#friction_M0_M2_trends = np.polyfit(meteo_years, friction_M0_M2_anomalies, deg=1)[0]

friction_M2_M2_anomalies = np.vstack([outputs[i]["friction_M2_M2_anomaly"] for i in range(len(meteo_years))])
#friction_M2_M2_trends = np.polyfit(meteo_years, friction_M2_M2_anomalies, deg=1)[0]

friction_M0_S2_anomalies = np.vstack([outputs[i]["friction_M0_S2_anomaly"] for i in range(len(meteo_years))])
#friction_M0_S2_trends = np.polyfit(meteo_years, friction_M0_S2_anomalies, deg=1)[0]

mask_subtidal = outputs[0]["mask_subtidal"].copy()

for d in outputs[1:]:
    mask_subtidal &= d["mask_subtidal"]

np.save('../../data/DCSM/mask_subtidal.npy', mask_subtidal)

# bound = np.nanpercentile(abs(friction_M2_M4_trends[mask_subtidal]), 95)
# spatial_map(lons[mask_subtidal], lats[mask_subtidal], friction_M2_M4_trends[mask_subtidal], vmin=-bound, vmax=bound, clabel='Friction M2 M4 trend [?/yr]', savename='friction_asymmetry_trend.png', cmap=cm.vik)

# bound = np.nanpercentile(abs(friction_M0_M2_trends[mask_subtidal]), 95)
# spatial_map(lons[mask_subtidal], lats[mask_subtidal], friction_M0_M2_trends[mask_subtidal], vmin=-bound, vmax=bound, clabel='Friction M0 M2 trend [?/yr]', cmap=cm.vik)

# bound = np.nanpercentile(abs(friction_M2_M2_trends[mask_subtidal]), 95)
# spatial_map(lons[mask_subtidal], lats[mask_subtidal], friction_M2_M2_trends[mask_subtidal], vmin=-bound, vmax=bound, clabel='Friction M0 M2 trend [?/yr]', cmap=cm.vik)

# bound = np.nanpercentile(abs(friction_M0_S2_trends[mask_subtidal]), 95)
# spatial_map(lons[mask_subtidal], lats[mask_subtidal], friction_M0_S2_trends[mask_subtidal], vmin=-bound, vmax=bound, clabel='Friction M0 M2 trend [?/yr]', cmap=cm.vik)




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

#%%
friction_M0_M2_2009 = np.nansum(outputs[13]['friction_M0_M2_2009'][mask])
friction_M2_M4_2009 = np.nansum(outputs[13]['friction_M2_M4_2009'][mask])
friction_M2_M2_2009 = np.nansum(outputs[13]['friction_M2_M2_2009'][mask])
friction_M0_S2_2009 = np.nansum(outputs[13]['friction_M0_S2_2009'][mask])
ts_friction_M0_M2_anomalies = np.array([np.nansum(outputs[i]['friction_M0_M2_anomaly'][mask]) for i in range(len(meteo_years))])
ts_friction_M2_M4_anomalies = np.array([np.nansum(outputs[i]['friction_M2_M4_anomaly'][mask]) for i in range(len(meteo_years))])
ts_friction_M2_M2_anomalies = np.array([np.nansum(outputs[i]['friction_M2_M2_anomaly'][mask]) for i in range(len(meteo_years))])
ts_friction_M0_S2_anomalies = np.array([np.nansum(outputs[i]['friction_M0_S2_anomaly'][mask]) for i in range(len(meteo_years))])
ts_friction_M0_M2 = ts_friction_M0_M2_anomalies + friction_M0_M2_2009
ts_friction_M2_M4 = ts_friction_M2_M4_anomalies + friction_M2_M4_2009
ts_friction_M2_M2 = ts_friction_M2_M2_anomalies + friction_M2_M2_2009
ts_friction_M0_S2 = ts_friction_M0_S2_anomalies + friction_M0_S2_2009


L = compute_estuary_length_between_cross_sections(6.5701, 53.5450, 6.6733, 53.58366, 7.078, 53.290, 7.078, 53.340)
enclosed_volume = np.array([np.nansum(outputs[i]['volume'][mask]) for i in range(len(bathy_years))])


ts_friction_M0_M2_anomalies = -ts_friction_M0_M2_anomalies / enclosed_volume * L * 1000
ts_friction_M2_M4_anomalies = -ts_friction_M2_M4_anomalies / enclosed_volume * L * 1000
ts_friction_M2_M2_anomalies = -ts_friction_M2_M2_anomalies / enclosed_volume * L * 1000
ts_friction_M0_S2_anomalies = -ts_friction_M0_S2_anomalies / enclosed_volume * L * 1000
ts_friction_M0_M2 = -ts_friction_M0_M2 / enclosed_volume * L * 1000
ts_friction_M2_M4 = -ts_friction_M2_M4 / enclosed_volume * L * 1000
ts_friction_M2_M2 = -ts_friction_M2_M2 / enclosed_volume * L * 1000
ts_friction_M0_S2 = -ts_friction_M0_S2 / enclosed_volume * L * 1000


plt.figure(figsize=(10, 4))
plt.plot(bathy_years, ts_friction_M0_M2, label='M0+M2')
plt.plot(bathy_years, ts_friction_M2_M4, label='M2+M4')
plt.plot(bathy_years, ts_friction_M2_M2, label='M2+M2')
plt.plot(bathy_years, ts_friction_M0_S2, label='M0+S2')
plt.legend()
plt.xlabel('Bathymetry year')
plt.ylabel('MSL contribution upstream [mm]')
plt.title('Friction')
#plt.savefig('../figures/friction_contributions.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
np.save('../../processed_time_series/DCSM/MSL_contributions/friction_M0_M2.npy', ts_friction_M0_M2)
np.save('../../processed_time_series/DCSM/MSL_contributions/friction_M2_M4.npy', ts_friction_M2_M4)
np.save('../../processed_time_series/DCSM/MSL_contributions/friction_M2_M2.npy', ts_friction_M2_M2)
np.save('../../processed_time_series/DCSM/MSL_contributions/friction_M0_S2.npy', ts_friction_M0_S2)

plt.figure(figsize=(10, 4))
plt.plot(bathy_years, ts_friction_M2_M4_anomalies, label='M2+M4')
plt.plot(bathy_years, ts_friction_M0_M2_anomalies, label='M0+M2', color='k')
plt.plot(bathy_years, ts_friction_M2_M2_anomalies, label='M2+M2')
plt.plot(bathy_years, ts_friction_M0_S2_anomalies, label='M0+S2')
plt.title('Friction')
plt.legend()
plt.xlabel('Bathymetry year')
plt.ylabel('MSL anomaly contribution upstream [mm]')
#plt.savefig('../../figures/friction_contributions.png', dpi=300, bbox_inches='tight', pad_inches=0.05)

def closest_N_idx(lons, lats, lon, lat, n):
    d2 = (lons - lon)**2 + (lats - lat)**2
    return np.argpartition(d2, n)[:n]

coordinates_knock = (7.030674, 53.327173)

N = 1
indices_knock = closest_N_idx(lons, lats, coordinates_knock[0], coordinates_knock[1], N)
indices_knock = indices_knock[mask_subtidal[indices_knock]]

mean_MSL_anomaly = np.mean(MSL_anomalies[:, indices_knock], axis=1)


friction_sum_anomalies = ts_friction_M0_M2_anomalies + ts_friction_M2_M4_anomalies #+ ts_friction_M0_S2# + ts_friction_M2_M2

plt.figure(figsize=(10, 4))
plt.plot(bathy_years, friction_sum_anomalies, label='Sum of M0+M2 and M2+M4 frictional contribution')
plt.plot(bathy_years, mean_MSL_anomaly, color='k', label='Modeled MSL anomaly')
plt.ylabel('MSL anomaly [mm]')
plt.xlabel('Bathymetry year')
plt.title('Mean sea level anomaly at Knock')
plt.legend()
#plt.savefig('../../figures/MSL_anomalies.png', dpi=300, bbox_inches='tight', pad_inches=0.05)