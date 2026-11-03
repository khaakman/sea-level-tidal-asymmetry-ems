#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 14})

min_percentile = 1
max_percentile = 55 

fn = '../../data/bathymetry/Quantile Regression-based Elevation Changes per Morph. Unit.xlsx'
sheet = 'EemsDollard'

def main():
    
    # Read Excel file (replace 'file.xlsx' with your filename)
    df = pd.read_excel(fn, sheet_name=sheet,
                       decimal=',',     # interpret commas as decimal points
                       index_col=0)     # use the first column (percentiles) as the index


    lower_bounds = df['Confidence Interval: Lower Bound'].values / 10
    upper_bounds = df['Confidence Interval: Upper Bound'].values / 10
    elevation_changes = df['Elevation Changes [mm/yr]'].values / 10
    idx = np.where(elevation_changes == np.max(elevation_changes))
    print(idx)
    print('percentile corresponding to max trend:', df.index[idx])
    xrange = 1.05 * np.nanmax(np.abs(lower_bounds))


    # Keep only columns that look like years (all digits)
    df = df.loc[:, df.columns.astype(str).str.fullmatch(r'\d{4}')]
    mean_depths = df.mean(axis=1)
    
    # Three subplots:
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Left plot spans both rows
    ax = fig.add_subplot(gs[:, 0])
    
    ax.plot(elevation_changes, mean_depths)
    ax.fill_betweenx(mean_depths, lower_bounds, upper_bounds, alpha=0.3)
    ax.axvline(0, color='k')
    ax.set_xlabel('Elevation trend [cm/yr]')
    ax.set_xlim(-xrange, xrange)
    ax.set_ylabel('Depth [m]')
    ax.set_title('Percentile trends')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
    
    # Secondary axis
    desired_indices = [1, 5, 10, 30, 50, 70, 90, 99]
    positions = [(np.abs(df.index - idx)).argmin() for idx in desired_indices]
    tick_positions = mean_depths.iloc[positions]
    
    secax = ax.twinx()
    secax.set_ylim(ax.get_ylim())
    secax.set_yticks(tick_positions)
    secax.set_yticklabels(desired_indices)
    secax.set_ylabel('Percentile')
    secax.grid(False)
    
    # Top-right plot
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.plot(df.columns.astype(int), df.loc[min_percentile], marker='o')
    ax_top.set_title(f'Percentile {min_percentile} time series')
    ax_top.set_ylabel('Depth [m]')
    ax_top.set_xticklabels('')
    ax_top.text(0.02, 0.98, '(b)', transform=ax_top.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')
    ax_top.set_ylim(top=-19.2)
    
    # Bottom-right plot (your second time series)
    ax_bottom = fig.add_subplot(gs[1, 1])
    ax_bottom.plot(df.columns.astype(int), df.loc[max_percentile], marker='o')
    ax_bottom.set_title(f'Percentile {max_percentile} time series')
    ax_bottom.set_xlabel('Year')
    ax_bottom.set_ylabel('Depth [m]')
    ax_bottom.text(0.02, 0.98, '(c)', transform=ax_bottom.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='left')
    
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.savefig('../../figures/Figure2.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    
    return






if __name__ == '__main__':
    main()
