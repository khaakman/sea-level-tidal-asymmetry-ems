#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def main():
    years = np.arange(1996, 2023)
    ts_friction_M0_M2_anomalies = np.load('../../processed_time_series/DCSM/MSL_contributions/friction_M0_M2_anomalies.npy', )
    ts_friction_M2_M4_anomalies = np.load('../../processed_time_series/DCSM/MSL_contributions/friction_M2_M4_anomalies.npy', )
    MSL_anomalies = np.load('../../processed_time_series/DCSM/MSL_anomalies.npy')
    lats = np.load('../../data/DCSM/lats.npy')
    mask_end = np.load('../../data/DCSM/mask_end.npy')
    friction_sum_anomalies = ts_friction_M0_M2_anomalies + ts_friction_M2_M4_anomalies 
    
    upper_channel_indices = np.where(mask_end & (lats > 53.32239583))[0]
    median_MSL_anomaly = np.median(MSL_anomalies[:, upper_channel_indices], axis=1)
    
    # Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(years, friction_sum_anomalies, label='Sum of M0+M2 and M2+M4 frictional contribution')
    plt.plot(years, median_MSL_anomaly, color='k', label='Modeled MSL anomaly')
    plt.ylabel('MSL anomaly [mm]')
    plt.xlabel('Bathymetry year')
    plt.title('Mean sea level anomaly at upstream boundary')
    plt.legend()
    plt.savefig('../../figures/Figure8.png', dpi=300, bbox_inches='tight', pad_inches=0.05)

if __name__ == '__main__':
    main()