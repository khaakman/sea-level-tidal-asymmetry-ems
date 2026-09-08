#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import cmocean
from matplotlib.tri import Triangulation

coordinates_ems = {
    'knock': (7.030674, 53.327173),
    'delfzijl': (6.933123, 53.326355),
    'emshorn': (6.841204, 53.493517),
    'dukegat': (6.926207, 53.433602),
    'borkumfischerbalje': (6.747895, 53.557436),
    'borkumsudstrand': (6.661416, 53.576854),
    'emdenneueseeschleuse': (7.186348, 53.336781),
    'pogum': (7.259801, 53.321351),
}

def hatch_insignificant_patches(ax, x, y, p_values, hatches=['xxxx']):
    tri = Triangulation(x, y)

    # compute max edge length per triangle to find phantom triangles
    xt, yt = x[tri.triangles], y[tri.triangles]
    edge1 = np.hypot(xt[:,0]-xt[:,1], yt[:,0]-yt[:,1])
    edge2 = np.hypot(xt[:,1]-xt[:,2], yt[:,1]-yt[:,2])
    edge3 = np.hypot(xt[:,2]-xt[:,0], yt[:,2]-yt[:,0])
    max_edge = np.maximum.reduce([edge1, edge2, edge3])
    
    # inspect histogram to pick a sensible threshold before masking
    threshold = np.percentile(max_edge, 90)  # tune this
    tri.set_mask(max_edge > threshold)
    
    ax.tricontourf(tri, p_values, levels=[0.05, 1], colors='none', hatches=hatches, 
                   transform=ccrs.PlateCarree(), zorder=10)
    
    return

def downsample_average(array, factor):
    """
    Downsamples a 2D array by averaging non-overlapping blocks.

    Parameters:
    - array: 2D NumPy array
    - factor: int or tuple of two ints (downsampling factor for rows and cols)

    Returns:
    - downsampled 2D NumPy array
    """
    if isinstance(factor, int):
        factor = (factor, factor)
    m, n = factor

    # Ensure array dimensions are divisible by the factor
    h, w = array.shape
    h_crop = h - (h % m)
    w_crop = w - (w % n)
    array_cropped = array[:h_crop, :w_crop]

    # Reshape and average
    reshaped = array_cropped.reshape(h_crop // m, m, w_crop // n, n)
    downsampled = reshaped.mean(axis=(1, 3))
    return downsampled


def plot_bathymetry_ems(ax, downsampling_factor=8):
    fn = '../../data/bathymetry/topography_2009_ger.tif'
    lon_min, lon_max = 6.2, 7.5
    lat_min, lat_max = 53.2, 54.0
    
    
    with rasterio.open(fn) as src:

        # 1. Transform bbox from EPSG:4326 → raster CRS
        bbox_r = transform_bounds(
            'EPSG:4326', src.crs,
            lon_min, lat_min, lon_max, lat_max
        )

        # 2. Read only the window inside raster CRS
        window = from_bounds(*bbox_r, src.transform)
        data = src.read(1, window=window)

        # 3. Get the affine transform for this window
        wt = src.window_transform(window)

        # 4. Build 2D row/column index grids
        nrows, ncols = data.shape
        cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))

        # 5. Compute X,Y coordinates in raster CRS using affine transform
        X = wt.c + cols * wt.a + rows * wt.b
        Y = wt.f + cols * wt.d + rows * wt.e

        # 6. Reproject pixel centers to lon/lat (returns 1D arrays)
        lon, lat = rasterio.warp.transform(
            src.crs, "EPSG:4326",
            X.flatten(), Y.flatten()
        )

        # 7. Reshape back to 2D grids matching the raster
        lon = np.array(lon).reshape(data.shape)
        lat = np.array(lat).reshape(data.shape)
        
        data = downsample_average(data, downsampling_factor)
        lon = downsample_average(lon, downsampling_factor)
        lat = downsample_average(lat, downsampling_factor)
    im = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        shading='auto',
        cmap=cmocean.cm.deep.reversed(), vmin=-20, vmax=2
    )
    return im

def spatial_map(x, y, z, fig=None, ax=None, vmin=None, vmax=None, cmap=None, alpha=None, make_clb=0, clabel='', title='', savename=''):
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
    #fig.canvas.callbacks.connect('pick_event', on_pick)
    if savename != '':
        fig.savefig('../../figures/{}'.format(savename), dpi=300, bbox_inches='tight', pad_inches=0.05)
    return fig, ax


def project_amplitude_direction(A_u, A_v, phi_u, phi_v, nx, ny):
    A_cos = A_u * np.cos(np.radians(phi_u)) * nx + A_v * np.cos(np.radians(phi_v)) * ny
    A_sin = A_u * np.sin(np.radians(phi_u)) * nx + A_v * np.sin(np.radians(phi_v)) * ny
    A_tot = np.sqrt(A_cos**2 + A_sin**2)
    phi_tot = np.rad2deg(np.arctan2(A_sin, A_cos))
    return A_tot, phi_tot

def cross_section_length(lons, lats, mask):
    R = 6371
    dlat = np.radians(lats[mask, None]) - np.radians(lats[mask])
    dlon = np.radians(lons[mask, None]) - np.radians(lons[mask])

    a = np.sin(dlat/2)**2 + np.cos(np.radians(lats[mask])[:, None]) * np.cos(np.radians(lats[mask])) * np.sin(dlon/2)**2
    dist = 2 * R * np.arcsin(np.sqrt(a))

    W = np.max(dist) * 1000

    ds = W / np.sum(mask) #distance between cross-section points
    return W, ds

def cross_section_normal(lons, lats):
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

    # Convert to UTM
    x, y = transformer.transform(lons, lats)
    
    points = np.column_stack([x, y])

    # center the data
    mean = points.mean(axis=0)
    centered = points - mean
    
    # PCA via SVD
    _, _, vh = np.linalg.svd(centered)
    
    direction = vh[0]  # unit vector along best-fit line
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    
    # Optional: visualize normal vector
    # plt.figure()
    # plt.scatter(x, y)
    # plt.scatter(mean[0], mean[1])
    # scale = 1000  # adjust depending on your coordinate scale
    # plt.quiver(mean[0], mean[1],
    #            normal[0]*scale, normal[1]*scale,
    #            angles='xy', scale_units='xy', scale=1)
    # plt.axis('equal')
    
    return normal



def project_residual_current_centerline(lons, lats, Umean, Vmean, perpendicular=False):
    # Load manually defined centerline
    fn_x = '../../data/centerline_x.npy'
    fn_y = '../../data/centerline_y.npy'
    centerline_lon = np.load(fn_x)
    centerline_lat = np.load(fn_y)
    
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

    # Convert to UTM
    centerline_x, centerline_y = transformer.transform(centerline_lon, centerline_lat)
    
    residual_currents = np.zeros(len(lons))
    
    for i in range(len(lons)):
        grid_lon, grid_lat = lons[i], lats[i]
        grid_x, grid_y = transformer.transform(grid_lon, grid_lat)
    
        dist_x = grid_x - centerline_x
        dist_y = grid_y - centerline_y
        distances = np.sqrt(dist_x**2 + dist_y**2)
        nearest_idx = np.argmin(distances)
        
        if nearest_idx == 0:
            # Forward difference at the first point
            dx = centerline_x[1] - centerline_x[0]
            dy = centerline_y[1] - centerline_y[0]
        elif nearest_idx == len(centerline_x) - 1:
            # Backward difference at the last point
            dx = centerline_x[-1] - centerline_x[-2]
            dy = centerline_y[-1] - centerline_y[-2]
        else:
            # Central difference for interior points
            dx = centerline_x[nearest_idx + 1] - centerline_x[nearest_idx - 1]
            dy = centerline_y[nearest_idx + 1] - centerline_y[nearest_idx - 1]

        # Normalize to get unit tangent vector
        magnitude = np.sqrt(dx**2 + dy**2)
        nx = dx / magnitude
        ny = dy / magnitude
        
        if not perpendicular:
            residual_currents[i] = Umean[i] * nx + Vmean[i] * ny
        else:
            residual_currents[i] = -Umean[i] * ny + Vmean[i] * nx
        
    
    return residual_currents

def project_amplitudes_centerline(lons, lats, A_U, A_V, phi_U, phi_V):
    # Load manually defined centerline
    fn_x = '../../data/centerline_x.npy'
    fn_y = '../../data/centerline_y.npy'
    centerline_lon = np.load(fn_x)
    centerline_lat = np.load(fn_y)
    
    # Convert phases to radians
    phi_U = np.deg2rad(phi_U)
    phi_V = np.deg2rad(phi_V)
    
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

    # Convert to UTM
    centerline_x, centerline_y = transformer.transform(centerline_lon, centerline_lat)

    A_tot = np.zeros(len(lons))
    phi_tot = np.zeros(len(lons))
    nx_array = np.zeros(len(lons))
    ny_array = np.zeros(len(lons))
    
    for i in range(len(lons)):
        grid_lon, grid_lat = lons[i], lats[i]
        grid_x, grid_y = transformer.transform(grid_lon, grid_lat)
    
        dist_x = grid_x - centerline_x
        dist_y = grid_y - centerline_y
        distances = np.sqrt(dist_x**2 + dist_y**2)
        nearest_idx = np.argmin(distances)
        
        if nearest_idx == 0:
            # Forward difference at the first point
            dx = centerline_x[1] - centerline_x[0]
            dy = centerline_y[1] - centerline_y[0]
        elif nearest_idx == len(centerline_x) - 1:
            # Backward difference at the last point
            dx = centerline_x[-1] - centerline_x[-2]
            dy = centerline_y[-1] - centerline_y[-2]
        else:
            # Central difference for interior points
            dx = centerline_x[nearest_idx + 1] - centerline_x[nearest_idx - 1]
            dy = centerline_y[nearest_idx + 1] - centerline_y[nearest_idx - 1]

        # Normalize to get unit tangent vector
        magnitude = np.sqrt(dx**2 + dy**2)
        nx = dx / magnitude
        ny = dy / magnitude
        
        nx_array[i] = nx
        ny_array[i] = ny
        
        A_cos = A_U[i] * np.cos(phi_U[i]) * nx + A_V[i] * np.cos(phi_V[i]) * ny
        A_sin = A_U[i] * np.sin(phi_U[i]) * nx + A_V[i] * np.sin(phi_V[i]) * ny
        A_tot[i] = np.sqrt(A_cos**2 + A_sin**2)
        
        phi_tot[i] = np.rad2deg(np.arctan2(A_sin, A_cos))
        
    return A_tot, phi_tot, nx_array, ny_array


def compute_estuary_length_between_cross_sections(lon0, lat0, lon1, lat1, lon2, lat2, lon3, lat3):
    fn_x = '../../data/centerline_x.npy'
    fn_y = '../../data/centerline_y.npy'
    
    centerline_lon = np.load(fn_x)
    centerline_lat = np.load(fn_y)
    
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

    # Convert to UTM
    x0, y0 = transformer.transform(lon0, lat0)
    x1, y1 = transformer.transform(lon1, lat1)
    x2, y2 = transformer.transform(lon2, lat2)
    x3, y3 = transformer.transform(lon3, lat3)
    centerline_x, centerline_y = transformer.transform(centerline_lon, centerline_lat)

    # --- cumulative distance along centerline ---
    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    seg_len = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(seg_len)))

    # --- find intersections WITH segment info ---
    i0, u0, p0 = segment_intersection_with_index(x0, y0, x1, y1, centerline_x, centerline_y)
    i1, u1, p1 = segment_intersection_with_index(x2, y2, x3, y3, centerline_x, centerline_y)

    # --- convert to chainage ---
    s0 = s[i0] + u0 * seg_len[i0]
    s1 = s[i1] + u1 * seg_len[i1]

    distance = abs(s1 - s0)
    
    # plt.figure()
    # plt.plot([x0, x1], [y0, y1])
    # plt.scatter(centerline_x, centerline_y)
    # plt.scatter(*p0, color='green')
    # plt.scatter(*p1, color='green')
    # plt.title('Distance: {}'.format(distance))
    return distance

def segment_intersection_with_index(x0, y0, x1, y1, cx, cy):
    p = np.array([x0, y0])
    r = np.array([x1 - x0, y1 - y0])

    def cross(a, b):
        return a[0]*b[1] - a[1]*b[0]

    for i in range(len(cx) - 1):
        q = np.array([cx[i], cy[i]])
        s = np.array([cx[i+1] - cx[i], cy[i+1] - cy[i]])

        denom = cross(r, s)
        if abs(denom) < 1e-12:
            continue

        qp = q - p
        t = cross(qp, s) / denom
        u = cross(qp, r) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p + t * r
            return i, u, intersection

    raise ValueError("No intersection found")
    return

def compute_phase_lag(phases, phase_lag_constits, const_list):
    phase_idx = np.zeros(len(phase_lag_constits))
    
    for i, c in enumerate(phase_lag_constits):
        phase_idx[i] = np.where(np.array(const_list) == phase_lag_constits[i])[0][0]
    
    if len(phase_lag_constits) == 2:
        phase_lag = (2*phases[int(phase_idx[0])] - phases[int(phase_idx[1])]) % 360
    if len(phase_lag_constits) == 3:
        phase_lag = (phases[int(phase_idx[0])] + phases[int(phase_idx[1])] - phases[int(phase_idx[2])]) % 360
    
    return phase_lag