#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 11:50:04 2025

@author: kbhaakman
"""

import matplotlib.pyplot as plt
import sys #to be able to load functions from paper_helper_functions.py inside analysis folder
sys.path.append("../analysis")
from paper_helper_functions import plot_bathymetry_ems, coordinates_ems
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
plt.style.use('ggplot')

stations = ['borkumsudstrand', 'borkumfischerbalje', 'emshorn', 'dukegat', 'knock', 'delfzijl', 'pogum'] 
names = ['Borkum Sudstrand', 'Borkum Fischerbalje', 'Emshorn', 'Dukegat', 'Knock', 'Delfzijl', 'Pogum']

# bounding box Bathymetry for reference:
# lon_min, lon_max = 6.2, 7.5
# lat_min, lat_max = 53.2, 54.0
extent = [6, 8, 53.1, 54.1]
lon_gridlines = FixedLocator([6.5, 7, 7.5])
lat_gridlines = FixedLocator([53.4, 53.8])


polygon_enclosed = [
    (6.5701, 53.5450),
    (6.69, 53.37),
    (7.078, 53.290),
    (7.078, 53.340),
    (7.00, 53.50),
    (6.6733, 53.58366)
]

poly_enclosed = Polygon(polygon_enclosed, edgecolor='red', facecolor='none', linewidth=2, transform=ccrs.PlateCarree())

def main():
    
    lons, lats = [], []
    PSMSL_lons, PSMSL_lats = [], []

    for i, station in enumerate(stations):
        lon, lat = coordinates_ems[station]
        
        if (station == 'delfzijl') or (station == 'borkumfischerbalje'):
            PSMSL_lons.append(lon)
            PSMSL_lats.append(lat)
        else:
            lons.append(lon)
            lats.append(lat)
        
    
    fig = plt.figure(figsize=(11.69,8.27))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(central_longitude=0))
    ax.scatter(lons, lats, c='cyan', s=100, marker='*', label='Not in PSMSL', transform=ccrs.PlateCarree(), zorder=10)
    ax.scatter(PSMSL_lons, PSMSL_lats, c='r', s=100, marker='*', label='PSMSL',  transform=ccrs.PlateCarree(), zorder=10)
    bathy_im = plot_bathymetry_ems(ax, downsampling_factor=8)
    ax.add_patch(poly_enclosed)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    #ax.coastlines()
    ax.legend(loc='lower left')
    
    dy = 0.015
    fs = 18
    ax.text(lons[0]-0.01, lats[0]+dy, s='1.', fontsize=fs, transform=ccrs.PlateCarree())
    ax.text(PSMSL_lons[0], PSMSL_lats[0]+dy, s='2.', fontsize=fs, transform=ccrs.PlateCarree())
    ax.text(lons[1], lats[1]+dy, s='3.', fontsize=fs, transform=ccrs.PlateCarree())
    ax.text(lons[2], lats[2]+dy, s='4.', fontsize=fs, transform=ccrs.PlateCarree())
    ax.text(lons[3], lats[3]+dy, s='5.', fontsize=fs, transform=ccrs.PlateCarree())
    ax.text(PSMSL_lons[1]-0.01, PSMSL_lats[1]+dy, s='6.', fontsize=fs, transform=ccrs.PlateCarree())
    ax.text(lons[4], lats[4]+dy, s='7.', fontsize=fs, transform=ccrs.PlateCarree())
    
        
        
        
    
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray')
    
    gl.top_labels = False
    gl.right_labels = False
    
    try:
        gl.xlocator = lon_gridlines
        gl.ylocator = lat_gridlines
    except NameError:
        pass
    
    

    # Make numbered multiline string
    text = "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))
    ax.text(
        0.75, 0.05,           # x, y position (slightly outside the axis)
        text,
        transform=ax.transAxes,
        fontsize=10,
        va='bottom',
        ha='left',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )
    
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    clb = plt.colorbar(bathy_im, cax=cax) 
    clb.set_label('Bathymetry [m]')
    
    plt.savefig('../../figures/tide_gauge_locations.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    return


if __name__ == '__main__':
    main()