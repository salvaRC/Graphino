"""
Author: Salva Rühling Cachay
"""

import os
import re
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xa


def cord_mask(data: xa.DataArray, is_flattened=False, flattened_too=False, lat=(-5, 5), lon=(190, 240)):
    """
    :param data:
    :param dim:
    :return:
    """
    oni_mask = {'time': slice(None), 'lat': slice(lat[0], lat[1]), 'lon': slice(lon[0], lon[1])}
    if flattened_too:
        flattened = data.copy() if is_flattened else data.stack(cord=['lat', 'lon']).copy()
        flattened_mask = ((lat[0] <= flattened.lat) & (flattened.lat <= lat[1]) & (lon[0] <= flattened.lon) & (flattened.lon <= lon[1]))
        # flattened[:, :] = 0
        # flattened.loc[oni_mask] = 1  # Masked (ONI) region has 1 as value
        # flattened_mask = (flattened[0, :] == 1)
        # print(np.count_nonzero(flattened_mask), '<<<<<<<<<<<<<<<<<')
        # flattened.sel(oni_mask) == flattened.loc[:, flattened_mask]
        return oni_mask, flattened_mask
    return oni_mask


def get_index_mask(data, index, flattened_too=False, is_data_flattened=False):
    """
    Get a mask to mask out the region used for  the ONI/El Nino3.4 or ICEN index.
    :param data:
    :param index: ONI or Nino3.4 or ICEN
    :return:
    """
    lats, lons = get_region_bounds(index)
    return cord_mask(data, lat=lats, lon=lons, flattened_too=flattened_too, is_flattened=is_data_flattened)


def get_region_bounds(index):
    if index.lower() in ["nino3.4", "oni"]:
        return (-5, 5), (190, 240)  # 170W-120W
    elif index.lower() == "icen":
        return (-10, 0), (270, 280)  # 90W-80W
    elif index.lower() in ["all", "world"]:
        return (-60, 60), (0, 360)  #
    else:
        raise ValueError("Unknown region/index")


def is_in_index_region(lat, lon, index="ONI"):
    lat_bounds, lon_bounds = get_region_bounds(index=index)
    if lat_bounds[0] <= lat <= lat_bounds[1]:
        if lon_bounds[0] <= lon <= lon_bounds[1]:
            return True
    return False


def get_euclidean_adj(data, radius_lat=3, radius_lon=3, self_loop=True):
    """

    :param data:
    :param radius_lat: degrees latitude that will be considered direct neighbors
    :param radius_lon: degrees longitude that will be considered direct neighbors
    :param self_loop: whether to add self-loops (default) or not
    :return:
    """
    n_nodes = len(data.indexes['cord'])
    adj = np.zeros((n_nodes, n_nodes))  # N x N
    tmp = xa.DataArray(adj, dims=("x1", "cord"), coords={"x1": range(n_nodes), "cord": data.indexes['cord']})
    for i in range(n_nodes):
        node = data.indexes["cord"][i]
        lat, lon = node[0], node[1]
        neighbors = {'lat': slice(lat - radius_lat, lat + radius_lat),
                     'lon': slice(lon - radius_lon, lon + radius_lon)}
        tmp.loc[i, neighbors] = 1

    assert np.count_nonzero(tmp.values != tmp.values.T) == 0  # symmetric adjacency matrix...
    matrix = tmp.values
    diagonals = 1 if self_loop else 0
    for i in range(n_nodes):
        matrix[i, i] = diagonals
    return matrix


def mask_adj_out(adj, max_distance, coordinates, return_xarray=False):
    """

    :param adj: Adjacency matrix
    :param max_distance: All edges larger than that distance are set to 0
    :param coordinates: list of semantic coordinates (lat_i, lon_i) of each node i
    :return:
    """
    n_nodes = adj.shape[0]
    assert n_nodes == adj.shape[1], "Adjacency matrix must be #Nodes x #Nodes"
    tmp = xa.DataArray(adj, dims=("x1", "cord"), coords={"x1": range(n_nodes), "cord": coordinates})
    new_adj = np.zeros(((n_nodes, n_nodes)))
    new_adj = xa.DataArray(new_adj, dims=("x1", "cord"), coords={"x1": range(n_nodes), "cord": coordinates})
    for i in range(n_nodes):
        node = coordinates[i]
        lat, lon = node[0], node[1]
        # Would be easier to negate the mask below, then no need for new_adj
        neighbors = {'lat': slice(lat - max_distance, lat + max_distance),
                     'lon': slice(lon - max_distance, lon + max_distance)}
        new_adj.loc[i, neighbors] = tmp.loc[i, neighbors]
    if not return_xarray:
        return new_adj.values
    return new_adj


def get_activation_function(name, functional=False, num=1, device='cuda'):
    name = name.lower().strip()
    funcs = {"softmax": F.softmax, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "identity": None,
             None: None, 'swish': F.silu, 'silu': F.silu, 'elu': F.elu,
             'prelu': nn.PReLU()}

    nn_funcs = {"softmax": nn.Softmax(dim=1), "relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(), 'silu': nn.SiLU(), 'elu': nn.ELU(), 'prelu': nn.PReLU(),
                'swish': nn.SiLU(), 'gelu': nn.GELU()}
    if num == 1:
        return funcs[name] if functional else nn_funcs[name]
    else:
        return [nn_funcs[name].to(device) for _ in range(num)]


def set_gpu(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def set_seed(seed, device='cuda'):
    # setting seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
