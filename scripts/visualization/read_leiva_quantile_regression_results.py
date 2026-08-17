#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 11:58:55 2025

@author: kbhaakman
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})

percentile = 1  # for example
#fn = '/Users/kbhaakman/Documents/PhD/tidal-asymmetry-trends/data/Quantile Regression-based Elevation Changes per Region.xlsx'
#sheet = 'Ems-Dollard'

fn = '../../data/bathymetry/Quantile Regression-based Elevation Changes per Morph. Unit.xlsx'
sheet = 'EemsDollard'

def main():
    
    # Read Excel file (replace 'file.xlsx' with your filename)
    df = pd.read_excel(fn, sheet_name=sheet,
                       decimal=',',     # interpret commas as decimal points
                       index_col=0)     # use the first column (percentiles) as the index

    print(df.head())
    print(df.columns)
    print(df.index)
    lower_bounds = df['Confidence Interval: Lower Bound'].values / 10
    upper_bounds = df['Confidence Interval: Upper Bound'].values / 10
    elevation_changes = df['Elevation Changes [mm/yr]'].values / 10
    xrange = 1.05 * np.nanmax(np.abs(lower_bounds))

    # Linear regression of quantiles (NOT THE SAME AS QUANTILE REGRESSION)
    until_col = 2015
    cols = df.columns[:df.columns.get_loc(until_col) + 1]

    x = np.arange(len(cols))
    df['trend_until_{}'.format(until_col)] = df[cols].apply(lambda row: linregress(x, row)[0], axis=1)
    elevation_changes2 = df['trend_until_{}'.format(until_col)].values * 100


    plt.figure()
    plt.scatter(elevation_changes, elevation_changes2)
    plt.title('Comparison quantile regression output vs linear regression of quantiles')
    plt.xlabel('Quantile regression trend [cm/yr]')
    plt.ylabel('Linear regression of quantiles trend [cm/yr]')

    # Keep only columns that look like years (all digits)
    df = df.loc[:, df.columns.astype(str).str.fullmatch(r'\d{4}')]
    mean_depths = df.mean(axis=1)
    print(mean_depths.iloc[-100:-80])

    plt.figure()
    plt.plot(elevation_changes, mean_depths)
    plt.fill_betweenx(mean_depths, lower_bounds, upper_bounds, alpha=0.3)
    plt.axvline(0, color='k')
    plt.xlabel('Elevation trend [cm/yr]')
    plt.xlim(-xrange, xrange)
    plt.ylabel('Depth [m]')
    #plt.savefig('../../figures/bathymetry_percentiles_trend.png', dpi=300, bbox_inches='tight', pad_inches=0.05)






    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(df.columns.astype(int), df.loc[percentile], marker='o')
    plt.title(f'Percentile {percentile} time series')
    plt.xlabel('Year')
    plt.ylabel('Depth [m]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #plt.savefig('../../figures/bathymetry_percentile_{}_trend.png'.format(percentile), dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    
    fig, axes = plt.subplots(figsize=(14,6), ncols=2)
    ax = axes[0]
    ax.plot(elevation_changes, mean_depths)
    ax.fill_betweenx(mean_depths, lower_bounds, upper_bounds, alpha=0.3)
    ax.axvline(0, color='k')
    ax.set_xlabel('Elevation trend [cm/yr]')
    ax.set_xlim(-xrange, xrange)
    ax.set_ylabel('Depth [m]')
    ax.set_title('Percentile trends')

    
    # Select a subset of ticks, e.g., every 5th point
    desired_indices = [1, 5, 10, 30, 50, 70, 90, 99]

    # Find corresponding mean_depths for these indices
    positions = [(np.abs(df.index - idx)).argmin() for idx in desired_indices]

    # Corresponding mean_depths at those positions
    tick_positions = mean_depths.iloc[positions]  # use iloc, not []
    
    # Secondary axis
    secax = ax.twinx()
    secax.set_ylim(ax.get_ylim())
    secax.set_yticks(tick_positions)
    secax.set_yticklabels(desired_indices)
    secax.set_ylabel('Percentile')
    secax.grid(False)
    
    ax = axes[1]
    ax.plot(df.columns.astype(int), df.loc[percentile], marker='o')
    ax.set_title(f'Percentile {percentile} time series')
    ax.set_xlabel('Year')
    ax.set_ylabel('Depth [m]')
    fig.subplots_adjust(wspace=0.5)
    fig.savefig('../../figures/bathymetry_trends.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    return






if __name__ == '__main__':
    main()
