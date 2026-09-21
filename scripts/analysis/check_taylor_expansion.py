#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:40:48 2026

@author: kbhaakman
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from paper_helper_functions import project_residual_current_centerline, project_amplitudes_centerline

fourier_folder = '../../data/DCSM/fourier_dynamic_meteo_dynamic_bathymetry'
meteo_yr = 2009
bathy_yr = 2009
partitions = [27, 28, 29]

constituents = {
    "M0": {"elev": 1, "u": 4, "v": 6, "residual":True},
    "M2": {"elev": 13, "u": 15, "v": 17},
    "M4": {"elev": 21, "u": 22, "v": 23},
    "S2": {"elev": 18, "u": 19, "v": 20},
    "M6": {"elev": 27, "u": 28, "v": 29},
    "MS4": {"elev": 24, "u": 25, "v": 26},
}

frequencies = {
    "M2": 2 * np.pi / (12.4206 * 3600),
    "S2": 2 * np.pi / (12.0000 * 3600),
    "M4": 2 * np.pi / (6.2103 * 3600),
    "M6": 2 * np.pi / (4.1402 * 3600),
    "MS4": 2 * np.pi / (6.1031 * 3600),
}

def main():
    actual_folder = fourier_folder + '/A04_meteo{}_bathy{}_fourier/'.format(meteo_yr, meteo_yr)
    
    out = {k: [] for k in [
        "lon", "lat", "depth", "min_depth",
        *[f"{c}_{x}" for c in constituents for x in
          ["elev_amp", "elev_phs", "U_amp", "U_phs", "u_amp", "u_phs", "v_amp", "v_phs"]]
            ]}

    for partition_nr in partitions:
        fn = f"{actual_folder}DCSM-FM_100m_00{partition_nr}_fou.nc"
        with xr.open_dataset(fn) as ds:
    
            lons = ds.mesh2d_face_x.values
            lats = ds.mesh2d_face_y.values
    
            out["lon"].append(lons)
            out["lat"].append(lats)
    
            bed_level = ds.mesh2d_flowelem_bl.values
            MSL = ds.mesh2d_fourier001_mean.values
            out["depth"].append(MSL - bed_level)
            out["min_depth"].append(ds.mesh2d_fourier002_min_depth.values)
    
            for constituent, indices in constituents.items():
    
                data = get_fourier_data(
                    ds, lons, lats, constituent, indices
                )
    
                for key, value in data.items():
                    out[key].append(value)
    for k in out:
        if len(out[k]) > 0:
            out[k] = np.concatenate(out[k])
            
    global t
    global eta
    global U
    global friction
    
    #t, eta, U = reconstruct_tides(out, constituents)
    t, eta, u, v, U = reconstruct_tides(out, constituents, frequencies)
    friction = U*np.abs(U) / (out['depth'] + eta)**(4/3)
    u_friction = u*np.sqrt(u**2 + v**2) / (out['depth'] + eta)**(4/3)
    v_friction =  v*np.sqrt(u**2 + v**2) / (out['depth'] + eta)**(4/3)
    
    mean_friction = np.mean(friction, axis=0)
    mean_u_friction = np.mean(u_friction, axis=0)
    mean_v_friction = np.mean(v_friction, axis=0)
    
    term1 = 1/out['depth']**(4/3) * np.mean(U * np.abs(U), axis=0)
    term2 = 4 / (3*out['depth']**(7/3)) * np.mean(eta * U * np.abs(U), axis=0)
    approx_friction = term1 - term2
    
    error = approx_friction - mean_friction
    rel_error = abs(error) / abs(mean_friction)
    norm_error = abs(error) / np.nanmedian(abs(mean_friction))
    eta_rms = np.sqrt(np.mean(eta**2, axis=0))
    frac = eta_rms / out['depth']
    
    
    spatial_map(out['lon'], out['lat'], frac, cmap='viridis', vmin=0, vmax=1, title=r'$\frac{\eta_{RMS}}{H}$')
    spatial_map(out['lon'], out['lat'], abs(error), cmap='viridis', vmin=0, vmax=0.01, title='Error in friction')
    spatial_map(out['lon'], out['lat'], rel_error, cmap='viridis', vmin=0, vmax=10, title='Rel. error in friction')
    spatial_map(out['lon'], out['lat'], norm_error, cmap='viridis', vmin=0, vmax=10, title='Rel. error in friction')
    
    spatial_map(out['lon'], out['lat'], out['depth'], cmap='viridis_r', title='mean depth', vmin=0, vmax=10)
    spatial_map(out['lon'], out['lat'], mean_friction, title='mean friction', cmap='bwr')
    
    mask = out['min_depth'] > 0.01
    #mask = (eta_rms / out['depth']) < 0.2
    
    total_proj_friction = np.nansum(mean_friction * out['depth'])
    total_u_friction = np.nansum(mean_u_friction * out['depth'])
    total_v_friction = np.nansum(mean_v_friction * out['depth'])
    total_proj_friction_mask = np.nansum(mean_friction[mask] * out['depth'][mask])
    total_u_friction_mask = np.nansum(mean_u_friction[mask] * out['depth'][mask])
    total_v_friction_mask = np.nansum(mean_v_friction[mask] * out['depth'][mask])
    print('Median friction per unit depth: ', np.nanmedian(abs(mean_friction)))
    print('Total mean friction:', round(total_proj_friction, 1), 'm$^{5/3} s^{-2}$')
    print('Total u friction:', round(total_u_friction, 1), 'm$^{5/3} s^{-2}$')
    print('Total v friction:', round(total_v_friction, 1), 'm$^{5/3} s^{-2}$')
    print('Total mean friction in mask:', round(total_proj_friction_mask, 1), 'm$^{5/3} s^{-2}$')
    print('Total u friction in mask:', round(total_u_friction_mask, 1), 'm$^{5/3} s^{-2}$')
    print('Total v friction in mask:', round(total_v_friction_mask, 1), 'm$^{5/3} s^{-2}$')
    x = frac
    y = rel_error
    finite_idx = np.isfinite(y) & mask
    x = x[finite_idx]
    y = y[finite_idx]
    
    err_threshold = 0.1
    n_below_err = (y < err_threshold).sum()
    pct = 100 * n_below_err / len(x)
    print(f"{pct:.1f}% of cells in mask have relative error below {err_threshold:.0%}")
    
    
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    print(len(x))
    A = np.ones((len(y), 2))
    A[:,1] = x**2
    xhat = np.linalg.solve(A.T @ A, A.T @ y)
    
    plt.figure()
    plt.scatter(x, 100*y)
    plt.plot(x, 100*A@xhat, 'r')
    plt.xlabel('$\eta_{RMS} / H$')
    plt.ylabel('Relative error [%]')
    


    return out

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

def on_pick(event):
    ind = event.ind[0]

    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    
    ax[0].plot(t / 3600, eta[:, ind])
    ax[0].set_ylabel("η [m]")
    
    ax[1].plot(t / 3600, U[:, ind])
    ax[1].set_ylabel("U [m/s]")
    
    
    
    ax[2].plot(t / 3600, friction[:, ind])
    ax[2].set_ylabel("friction [m$^{2/3}/s^2$]")
    ax[2].set_xlabel("Time [hours]")
    ax[2].set_title(np.mean(friction[:,ind]))
    
    plt.show()

    return

def get_fourier_data(ds, lons, lats, constituent, indices):
    """Extract raw and projected Fourier data."""

    def get(field, component):
        idx = indices[component]
        return getattr(ds, f"mesh2d_fourier{idx:03d}_{field}").values

    result = {}

    # Elevation
    if "elev" in indices:
        field = "mean" if indices.get("residual") else "amp"
    
        result["elev_amp"] = get(field, "elev")
    
        if not indices.get("residual"):
            result["elev_phs"] = get("phs", "elev")

    # Velocity
    if "u" in indices and "v" in indices:

        field = "mean" if indices.get("residual") else "amp"
        u_amp, v_amp = get(field, "u"), get(field, "v")

        result["u_amp"] = u_amp
        result["v_amp"] = v_amp

        if indices.get("residual"):
            result["U_amp"] = project_residual_current_centerline(
                lons, lats, u_amp, v_amp
            )
        else:
            u_phs = get("phs", "u")
            v_phs = get("phs", "v")

            result["u_phs"] = u_phs
            result["v_phs"] = v_phs

            U_amp, U_phs, _, _ = project_amplitudes_centerline(
                lons, lats, u_amp, v_amp, u_phs, v_phs
            )

            result["U_amp"] = U_amp
            result["U_phs"] = U_phs

    return {f"{constituent}_{k}": v for k, v in result.items()}

def reconstruct_tides(out, constituents, frequencies, n_times=1000):

    T_sn = 14.77 * 24 * 3600  # Spring-neap cycle [s]
    t = np.linspace(0, T_sn, n_times)

    n_faces = len(out["lon"])

    eta = np.zeros((n_times, n_faces))
    u = np.zeros_like(eta)
    v = np.zeros_like(eta)
    U = np.zeros_like(eta)

    for constituent, indices in constituents.items():

        # M0: mean contribution
        if indices.get("residual"):

            eta += out[f"{constituent}_elev_amp"][None, :]
            u += out[f"{constituent}_u_amp"][None, :]
            v += out[f"{constituent}_v_amp"][None, :]
            U += out[f"{constituent}_U_amp"][None, :]

            continue

        omega = frequencies[constituent]

        # Elevation
        amp_key = f"{constituent}_elev_amp"
        phs_key = f"{constituent}_elev_phs"

        if amp_key in out and phs_key in out:

            amp = out[amp_key]
            phs = np.deg2rad(out[phs_key])

            eta += amp[None, :] * np.cos(
                omega * t[:, None] - phs[None, :]
            )

        # Velocity components
        for component in ["u", "v"]:

            amp = out[f"{constituent}_{component}_amp"]
            phs = np.deg2rad(out[f"{constituent}_{component}_phs"])

            velocity = amp[None, :] * np.cos(
                omega * t[:, None] - phs[None, :]
            )

            if component == "u":
                u += velocity
            else:
                v += velocity

        # Projected velocity
        amp = out[f"{constituent}_U_amp"]
        phs = np.deg2rad(out[f"{constituent}_U_phs"])

        U += amp[None, :] * np.cos(
            omega * t[:, None] - phs[None, :]
        )

    return t, eta, u, v, U



if __name__ == '__main__':
    out = main()