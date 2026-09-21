#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
plt.style.use('ggplot')

colors = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
]

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


bathy_years = np.arange(1996, 2023)

ts_friction_M0_M2 = np.load('../../processed_time_series/DCSM/MSL_contributions/friction_M0_M2.npy')
ts_friction_M2_M2 = np.load('../../processed_time_series/DCSM/MSL_contributions/friction_M2_M2.npy')
ts_friction_M2_M4 = np.load('../../processed_time_series/DCSM/MSL_contributions/friction_M2_M4.npy')
ts_friction_M0_S2 = np.load('../../processed_time_series/DCSM/MSL_contributions/friction_M0_S2.npy')
ts_advection_M0 = np.load('../../processed_time_series/DCSM/MSL_contributions/advection_M0.npy')
ts_advection_M2 = np.load('../../processed_time_series/DCSM/MSL_contributions/advection_M2.npy')
ts_advection_M4 = np.load('../../processed_time_series/DCSM/MSL_contributions/advection_M4.npy')
ts_advection_S2 = np.load('../../processed_time_series/DCSM/MSL_contributions/advection_S2.npy')


fig, axes = plt.subplots(figsize=(10, 8), nrows=2)
ax = axes[0]
ax.plot(bathy_years, ts_friction_M0_M2, label='M0+M2')
ax.plot(bathy_years, ts_friction_M2_M4, label='M2+M4')
ax.plot(bathy_years, ts_friction_M2_M2, label='M2+M2')
ax.plot(bathy_years, ts_friction_M0_S2, label='M0+S2')
ax.legend()
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
#ax.set_ylabel('MSL contribution upstream [mm]')
ax.set_title('Friction')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

ax = axes[1]
ax.plot(bathy_years, ts_advection_M0, label='M0')
ax.plot(bathy_years, ts_advection_M2, label='M2')
ax.plot(bathy_years, ts_advection_M4, label='M4')
ax.plot(bathy_years, ts_advection_S2, label='S2')
ax.legend()
ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax.set_xlabel('Bathymetry year')
#ax.set_ylabel('MSL contribution upstream [mm]')
ax.set_title('Advection')
ax.set_ylim(top=35)

fig.supylabel('MSL contribution upstream [mm]', x=0.06)
fig.savefig('../../figures/Figure7.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
