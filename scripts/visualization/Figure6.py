#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:28:46 2026

@author: kbhaakman
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('ggplot')
import cmcrameri.cm as cm
from scipy.stats import t
import sys
sys.path.append("../analysis") #to be able to load dictionary from paper_helper_functions.py
from paper_helper_functions import hatch_insignificant_patches
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def main():
    global years
    global mask_subtidal
    global MSL_anomalies
    
    lons = np.load('../../data/DCSM/lons.npy')
    lats = np.load('../../data/DCSM/lats.npy')
    mask_subtidal = np.load('../../data/DCSM/mask_subtidal.npy')
    MSL_anomalies = np.load('../../processed_time_series/DCSM/MSL_anomalies.npy')
    years = np.arange(1996, 2023)
    coeffs, cov = np.polyfit(years, MSL_anomalies, deg=1, cov=True)


    # Slope and intercept
    MSL_trends = coeffs[0, :]
    MSL_trends_unc = np.sqrt(cov[0, 0, :])

    df = len(years) - 2 #degrees of freedom

    t_stat = MSL_trends / MSL_trends_unc
    # Two-sided p-value
    p_values = 2 * t.sf(np.abs(t_stat), df=df)
    #significant = p_values < 0.05
    #mask_subtidal &= significant
    
    min_trend = np.nanmin(MSL_trends[mask_subtidal])
    median_trend = np.nanmedian(MSL_trends[mask_subtidal])
    q25, q75 = np.nanpercentile(MSL_trends[mask_subtidal], [25, 75])
    max_trend = np.nanmax(MSL_trends[mask_subtidal])
    print(q25*26, median_trend*26, q75*26, 'mm')
    
    
    bound = np.nanpercentile(abs(MSL_trends[mask_subtidal]), 95)
    spatial_map(lons[mask_subtidal], lats[mask_subtidal], MSL_trends[mask_subtidal], p_values=p_values[mask_subtidal], vmin=-bound, vmax=bound, clabel='MSL trend [mm/yr]', savename='Figure6.png', cmap=cm.vik)
    bound_unc = np.nanpercentile(abs(MSL_trends_unc[mask_subtidal]), 95)
    spatial_map(lons[mask_subtidal], lats[mask_subtidal], MSL_trends_unc[mask_subtidal], vmin=0, vmax=bound_unc, clabel='MSL trend uncertainty [mm/yr]', savename='Figure6_std.png', cmap='viridis')
    return


def on_pick(event):
    ind = event.ind[0]
    col = np.flatnonzero(mask_subtidal)[ind]

    MSL = MSL_anomalies[:,col]
    
    
    
    plt.figure()
    plt.plot(years, MSL, c='k')
    plt.xlabel('Time')
    plt.ylabel('$\Delta$MSL w.r.t. 2009 bathymetry [mm]')
    plt.title('Grid point {}'.format(col))
    
    A = np.ones((len(years),2))
    A[:,0] = years
    xhat = np.linalg.solve(A.T @ A, A.T @ MSL)
    residuals = A@xhat - MSL
    plot_acf(residuals, lags=10, alpha=0.05, zero=True)
    
    N = len(residuals)
    acf_values = acf(residuals, nlags=10, fft=False)
    Neff = N / (1 + 2 * np.sum(acf_values[1:]))

    print(f"N = {N}")
    print(f"Effective sample size = {Neff:.1f}")

    return

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
    if p_values is not None:
        hatch_insignificant_patches(ax, x, y, p_values)

    
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

if __name__ == '__main__':
    main()
