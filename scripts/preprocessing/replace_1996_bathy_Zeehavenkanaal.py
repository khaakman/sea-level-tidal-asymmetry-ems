import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np
import cmocean

lon_min_bbox, lon_max_bbox = 6.6, 7.7
lat_min_bbox, lat_max_bbox = 53.1, 53.6
lon_min_bbox, lon_max_bbox = -180, 180
lat_min_bbox, lat_max_bbox = -90, 90
vmin = -10
vmax = 0
save_results = False

def load_DCSM_bathymetry(fn, lon_min_bbox, lat_min_bbox, lon_max_bbox, lat_max_bbox):
    data = xr.open_dataset(fn)
    lon_DCSM = data.NetNode_x.values
    lat_DCSM = data.NetNode_y.values
    bathymetry_DCSM = data.NetNode_z.values

    mask = (lon_DCSM >= lon_min_bbox) & (lon_DCSM <= lon_max_bbox) & \
           (lat_DCSM >= lat_min_bbox) & (lat_DCSM <= lat_max_bbox)

    return lon_DCSM[mask], lat_DCSM[mask], bathymetry_DCSM[mask], data, mask


year = 1996
fn = f'../../data/DCSM_yearly_bathymetry/DCSM-FM_100m_grid_20200130_depth_Ems_{year}.nc'
lon_DCSM, lat_DCSM, bathymetry_1996, data_1996, mask = load_DCSM_bathymetry(fn, lon_min_bbox, lat_min_bbox, lon_max_bbox, lat_max_bbox)

# --------------------------
# LASSO SELECTOR IMPLEMENTATION
# --------------------------

fig, ax = plt.subplots(figsize=(8, 6))

pts = ax.scatter(
    lon_DCSM, lat_DCSM,
    c=bathymetry_1996, s=1,
    vmin=vmin, vmax=vmax,
    cmap=cmocean.cm.deep.reversed()
)

plt.colorbar(pts)
plt.title(f'Interpolated bathymetry {year}')

selected_indices = []   # will store final indices
xy = np.column_stack((lon_DCSM, lat_DCSM))   # point pairs for vectorized tests


def onselect(verts):
    """
    verts = list of (x,y) vertices of the polygon drawn by lasso
    """
    global selected_indices

    poly_path = Path(verts)
    inside = poly_path.contains_points(xy)

    selected_indices = np.where(inside)[0].tolist()

    print("\n---------------------------------------------")
    print(f"Selected {len(selected_indices)} points")
    print("Indices:")
    print(selected_indices[:50], "..." if len(selected_indices) > 50 else "")
    print("---------------------------------------------\n")
    
    
    plt.figure() 
    im = plt.scatter(lon_DCSM[selected_indices], lat_DCSM[selected_indices], c=bathymetry_1996[selected_indices], s=1, vmax=vmax, vmin=vmin, cmap=cmocean.cm.deep.reversed()) 
    plt.colorbar(im)
    
    fn_1997 = '../../data/DCSM_yearly_bathymetry/DCSM-FM_100m_grid_20200130_depth_Ems_1997.nc'
    _, _, bathymetry_1997, data_1997, _ = load_DCSM_bathymetry(fn_1997, lon_min_bbox, lat_min_bbox, lon_max_bbox, lat_max_bbox)
    bathymetry_1996[selected_indices] = bathymetry_1997[selected_indices]
    data_1996.NetNode_z.values = bathymetry_1996
    if save_results:
        data_1996.to_netcdf('../../data/DCSM_yearly_bathymetry/DCSM-FM_100m_grid_20200130_depth_Ems_1996.nc')
        #np.save('Zeehavenkanaal_indices.npy', selected_indices)


lasso = LassoSelector(ax, onselect)

plt.show()
