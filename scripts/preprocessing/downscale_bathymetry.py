import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping, box
from shapely.ops import transform as shp_transform
from pyproj import Transformer
import warnings
import shutil


for year in np.arange(1996, 2023):
    # ------------------ USER SETTINGS ------------------
    mesh_nc = "../../data/DCSM_original_bathymetry/DCSM-FM_100m_grid_20200130_depth_20220725_net.nc"
    bathy_tif = "../../data/bathymetry/topography_{}_ger.tif".format(year)
    new_file = "../../data/DCSM_yearly_bathymetry/DCSM-FM_100m_grid_20200130_depth_Ems_{}.nc".format(year)
    save_output = False  # True to save to new NetCDF, False to keep in memory
    
    
    # bounding box in lon/lat (EPSG:4326)
    lon_min, lon_max = 6.2, 7.5
    lat_min, lat_max = 53.2, 54.0
    # ---------------------------------------------------
    
    print('### {} ###'.format(year))
    # Open mesh (copy if saving)
    if save_output:
        shutil.copy(mesh_nc, new_file)
        ds = Dataset(new_file, "r+")
    else:
        ds = Dataset(mesh_nc, "r")
    
    # read node lon/lat and connectivity
    node_lon = ds["NetNode_x"][:]    # EPSG:4326 lon
    node_lat = ds["NetNode_y"][:]    # EPSG:4326 lat
    var_elem = ds["NetElemNode"]
    orig_elem_nodes = var_elem[:]    # shape (nElem, nMaxNode)
    fillv = getattr(var_elem, "_FillValue", None)
    start_index = int(getattr(var_elem, "start_index", 1))
    
    # build zero-based connectivity with -1 for padded entries
    elem_nodes0 = orig_elem_nodes.astype(int) - start_index
    if fillv is not None:
        elem_nodes0[orig_elem_nodes == fillv] = -1
    
    nElem = elem_nodes0.shape[0]
    nNode = node_lon.size

    
    # mask for nodes inside the lon/lat bbox (we will only update these)
    nodes_in_bbox = (
        (node_lon >= lon_min) & (node_lon <= lon_max) &
        (node_lat >= lat_min) & (node_lat <= lat_max)
    )
    
    # make bbox polygon in EPSG:4326 and transform it to EPSG:25832 (raster CRS)
    bbox_4326 = box(lon_min, lat_min, lon_max, lat_max)
    transformer_4326_to_25832 = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    bbox_25832 = shp_transform(transformer_4326_to_25832.transform, bbox_4326)
    
    # reproject all node coordinates to EPSG:25832 (vectorized)
    x_m, y_m = transformer_4326_to_25832.transform(node_lon, node_lat)  # arrays
    
    # open raster (EPSG:25832)
    rsrc = rasterio.open(bathy_tif)
    
    # arrays to hold element mean and intersection area (in m^2)
    elem_mean = np.full(nElem, np.nan)
    elem_area = np.zeros(nElem, dtype=float)  # area of the intersection (m^2)
    
    # adjacency list: elements touching each node (we'll need it for nodal averaging)
    nodes_adj_elems = [[] for _ in range(nNode)]
    
    # loop elements (safe, but may be slow for millions of elements)
    for i in range(nElem):
        inds = elem_nodes0[i]
        valid = inds[inds >= 0].astype(int)
        if valid.size < 3:
            continue
    
        # polygon in raster CRS (EPSG:25832)
        coords = [(float(x_m[j]), float(y_m[j])) for j in valid]
        poly = Polygon(coords)
        if not poly.is_valid or poly.is_empty or poly.area == 0.0:
            # try to fix geometry
            poly = poly.buffer(0)
            if poly.is_empty or poly.area == 0.0:
                continue
    
        # only consider the piece inside the lon/lat bbox (converted to EPSG:25832)
        poly_clip = poly.intersection(bbox_25832)
        if poly_clip.is_empty:
            # do not set elem_mean (remains NaN); no adjacency changes needed
            continue
    
        # store intersection area (used later for weighting)
        elem_area[i] = float(poly_clip.area)
    
        # build adjacency (we add element to each node even if only partially overlapped;
        # later we only update nodes inside bbox)
        for n in valid:
            nodes_adj_elems[n].append(i)
    
        # sample raster within the clipped polygon
        try:
            geojson = [mapping(poly_clip)]
            out_image, out_transform = mask(rsrc, geojson, crop=True, all_touched=True, nodata=rsrc.nodata)
            
            data = out_image[0]
    
            data[data < -2000] = np.nan  # Remove faulty values
            if rsrc.nodata is not None:
                data = np.ma.array(data, mask=(data == rsrc.nodata))
            else:
                data = np.ma.masked_invalid(data)
            if data.count() > 0:
                elem_mean[i] = float(data.mean())
            else:
                # intersection overlaps raster but raster has only nodata here
                elem_mean[i] = np.nan
            
        except ValueError:
            # rasterio raises ValueError if shapes do not overlap raster; treat as no-data
            elem_mean[i] = np.nan
        except Exception as e:
            warnings.warn(f"Element {i} mask error: {e}")
            elem_mean[i] = np.nan
            
        # Count valid raster values
        finite_mask = np.isfinite(out_image[0])
        finite_pixels = np.count_nonzero(finite_mask)
        total_pixels = out_image[0].size
        
        # Require a high percentage of valid bathymetry inside element
        valid_fraction = finite_pixels / total_pixels
        if valid_fraction < 0.9:  # Require at least 90% valid samples
            elem_mean[i] = np.nan
            elem_area[i] = 0.0
            continue  # Skip this element
            
    
    # Now compute nodal bathymetry (area-weighted average of adjacent element means)
    node_bathy = np.full(nNode, np.nan)
    
    for node_idx in range(nNode):
        elems = nodes_adj_elems[node_idx]
        if not elems:
            continue
        # pick only elements that have finite mean and positive intersection area
        elems = np.array(elems, dtype=int)
        means = elem_mean[elems]
        areas = elem_area[elems]
        valid_mask = np.isfinite(means) & (areas > 0)
        if not valid_mask.any():
            continue
        vals = means[valid_mask]
        w = areas[valid_mask]
        node_bathy[node_idx] = np.sum(w * vals) / np.sum(w)
    
    # Merge with original NetNode_z: keep original where:
    #  - node is outside the lon/lat bbox OR
    #  - the computed node_bathy is NaN
    orig_depth = None
    if "NetNode_z" in ds.variables:
        orig_depth = ds["NetNode_z"][:].astype(float)
    else:
        orig_depth = np.full(nNode, np.nan)
    
    updated = orig_depth.copy()
    
    # update only nodes inside bbox AND with a finite computed value
    mask_update = nodes_in_bbox & np.isfinite(node_bathy)
    updated[mask_update] = node_bathy[mask_update]
    
    # If saving, write updated NetNode_z into dataset
    if save_output:
        if "NetNode_z" in ds.variables:
            ds["NetNode_z"][:] = updated
        else:
            var = ds.createVariable("NetNode_z", "f8", ("nNetNode",))
            var.units = "m"
            var.standard_name = "sea_floor_depth"
            var.positive = "up"
            var.long_name = "bed level at net nodes (flow element corners)"
            ds["NetNode_z"][:] = updated
    
    # cleanup
    ds.close()
    rsrc.close()
    
    # summary
    n_updated = int(np.count_nonzero(mask_update))
    print(f"Elements processed (computed intersection): {np.count_nonzero(elem_area>0)} / {nElem}")
    print(f"Nodes inside bbox: {int(np.count_nonzero(nodes_in_bbox))}")
    print(f"Nodes updated (inside bbox & finite): {n_updated}")
    if save_output:
        print(f"Saved updated NetNode_z to {new_file}")
    else:
        print("Finished (no file written, save_output=False).")
