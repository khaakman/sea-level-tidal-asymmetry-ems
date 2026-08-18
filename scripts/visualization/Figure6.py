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

years = np.arange(1996, 2023)
lons = np.load('../../data/DCSM/lons.npy')
lats = np.load('../../data/DCSM/lats.npy')
mask_subtidal = np.load('../../data/DCSM/mask_subtidal.npy')
MSL_anomalies = np.load('../../processed_time_series/DCSM/MSL_anomalies.npy')
MSL_trends = np.polyfit(years, MSL_anomalies, deg=1)[0]

def main():
    bound = np.nanpercentile(abs(MSL_trends[mask_subtidal]), 95)
    spatial_map(lons[mask_subtidal], lats[mask_subtidal], MSL_trends[mask_subtidal], vmin=-bound, vmax=bound, clabel='MSL trend [mm/yr]', savename='Figure6.png', cmap=cm.vik)
    
    return


def on_pick(event):
    ind = event.ind[0]
    col = np.flatnonzero(mask_subtidal)[ind]

    MSL = 1000 * MSL_anomalies[:,col]
    
    plt.figure()
    plt.plot(years, MSL, c='k')
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

if __name__ == '__main__':
    main()
