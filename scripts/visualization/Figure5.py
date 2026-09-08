#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cmcrameri.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import t
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append("../analysis") #to be able to load dictionary from paper_helper_functions.py
from paper_helper_functions import hatch_insignificant_patches
plt.style.use('ggplot')

plot_intertidal_areas = False

def main():

    M2_M4_U_phase_2009_deg = np.load('../../processed_time_series/DCSM/harmonic_analysis/2M2_M4_U_phase_2009.npy')
    M2_M4_U_phase_2009_cos = np.cos(np.radians(M2_M4_U_phase_2009_deg))
    ts_2M2_M4_U_phase_anomalies = np.load('../../processed_time_series/DCSM/harmonic_analysis/2M2_M4_U_phase_anomalies.npy')
    
    lons = np.load('../../data/DCSM/lons.npy')
    lats = np.load('../../data/DCSM/lats.npy')
    subtidal_mask = np.load('../../data/DCSM/mask_subtidal.npy')
    bathy_years = np.arange(1996, 2023)
    
    
    ts_2M2_M4_U_phase_wrt_2009_cos = np.cos(np.radians(ts_2M2_M4_U_phase_anomalies + M2_M4_U_phase_2009_deg))
    coeffs, cov = np.polyfit(bathy_years, ts_2M2_M4_U_phase_wrt_2009_cos, deg=1, cov=True)
    M2_M4_U_phase_trend = coeffs[0,:]
    M2_M4_U_phase_trend_unc = np.sqrt(cov[0, 0, :])
    
    mask = np.isfinite(M2_M4_U_phase_trend)
    lons = lons[mask]
    lats = lats[mask]
    subtidal_mask = subtidal_mask[mask]
    M2_M4_U_phase_trend = M2_M4_U_phase_trend[mask]
    M2_M4_U_phase_trend_unc = M2_M4_U_phase_trend_unc[mask]
    M2_M4_U_phase_2009_cos = M2_M4_U_phase_2009_cos[mask]
    
    df = len(bathy_years) - 2 #degrees of freedom
    t_stat = M2_M4_U_phase_trend / M2_M4_U_phase_trend_unc
    DCSM_p_values = 2 * t.sf(np.abs(t_stat), df=df) # Two-sided p-value
    
    fig = plt.figure(figsize=(10, 10))
    bound = 1
    ax = fig.add_subplot(2, 1, 1, projection=ccrs.Mercator())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.set_extent([6.5, 7.6, 53.2, 53.7], crs=ccrs.PlateCarree())
    im = ax.scatter(lons[subtidal_mask], lats[subtidal_mask], c=M2_M4_U_phase_2009_cos[subtidal_mask], vmin=-bound, vmax=bound, cmap=cm.vik, s=1, transform=ccrs.PlateCarree(),)
    if plot_intertidal_areas:
        ax.scatter(lons[~subtidal_mask], lats[~subtidal_mask], c=M2_M4_U_phase_2009_cos[~subtidal_mask], vmin=-bound, vmax=bound, cmap=cm.vik, s=1, transform=ccrs.PlateCarree(), alpha=0.03)
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
    clb = plt.colorbar(im, cax=cax)
    clb.ax.tick_params(labelsize=10)
    clb.set_label('$\cos(2\phi^U_{M2} - \phi^U_{M4})$ [-]', fontsize=10)
    ax.text(0.02, 0.02, '(a)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='left')
    
    bound = np.nanpercentile(np.abs(M2_M4_U_phase_trend[subtidal_mask]), 95)
    ax = fig.add_subplot(2, 1, 2, projection=ccrs.Mercator())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.set_extent([6.5, 7.6, 53.2, 53.7], crs=ccrs.PlateCarree())
    im = ax.scatter(lons[subtidal_mask], lats[subtidal_mask], c=M2_M4_U_phase_trend[subtidal_mask], vmin=-bound, vmax=bound, cmap=cm.vik, s=1, transform=ccrs.PlateCarree())
    if plot_intertidal_areas:
        ax.scatter(lons[~subtidal_mask], lats[~subtidal_mask], c=M2_M4_U_phase_trend[~subtidal_mask],  vmin=-bound, vmax=bound, cmap=cm.vik, s=1, transform=ccrs.PlateCarree(),alpha=0.03)
    hatch_insignificant_patches(ax, lons[subtidal_mask], lats[subtidal_mask], DCSM_p_values[subtidal_mask])
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
    clb = plt.colorbar(im, cax=cax)
    clb.ax.tick_params(labelsize=10)
    clb.set_label('$\cos(2\phi^U_{M2} - \phi^U_{M4})$ trend [1/yr]', fontsize=10)
    ax.text(0.02, 0.02, '(b)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='left')
    
    
    fig.savefig('../../figures/Figure5.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    return

if __name__ == '__main__':
    main()
