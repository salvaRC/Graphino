"""
Author: Salva Rühling Cachay
"""

import numpy as np
import pandas as pd
import xarray as xa
import torch
from torch.utils.data import DataLoader

from utilities.utils import get_index_mask


class ENSO_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, labels):
        self.X = torch.tensor(X).float()
        self.labels = torch.tensor(labels).float()

    def __getitem__(self, i):
        return self.X[i], self.labels[i]

    def __len__(self):
        return self.X.shape[0]


def to_dataloaders(cmip5, soda, godas, batch_size, valid_split=0, validation='SODA', verbose=True,
                   concat_cmip5_and_soda=True, shuffle_training=True):
    """
     n - length of time series (i.e. dataset size)
     m - number of nodes/grid cells (33 if using exactly the ONI region)
    """

    sodaX = np.array(soda[0]) if not isinstance(soda[0], np.ndarray) else soda[0]
    cmip5X = np.array(cmip5[0]) if not isinstance(cmip5[0], np.ndarray) else cmip5[0]
    godasX = np.array(godas[0]) if not isinstance(godas[0], np.ndarray) else godas[0]

    if concat_cmip5_and_soda:  # instead of transfer, concat the cmip5 and soda data
        if validation.lower() == 'cmip5':
            first_val = min(len(godas[1]) * 2, 600)
            cmip5_trainX, cmip5_trainY = cmip5X[:-first_val], cmip5[1][:-first_val]
            validX, validY = cmip5X[-first_val:], cmip5[1][-first_val:]
            soda_trainX, soda_trainY = sodaX, soda[1]
        elif 'soda' in validation.lower():
            cmip5_trainX, cmip5_trainY = cmip5X, cmip5[1]
            if valid_split > 0:
                first_val = int(valid_split * len(sodaX))
                soda_trainX, soda_trainY = sodaX[:-first_val], soda[1][:-first_val]
                if validation.lower() == 'soda':
                    validX, validY = sodaX[-first_val:], soda[1][-first_val:]
                else:
                    validX, validY = sodaX, soda[1]
            else:  # without val. set, just return the SODA set
                soda_trainX, soda_trainY = sodaX, soda[1]
                validX, validY = sodaX, soda[1]
        else:
            raise ValueError('Validation dataset must be either CMIP5 or SODA')
        trainX = np.concatenate((cmip5_trainX, soda_trainX), axis=0)
        trainY = np.concatenate((cmip5_trainY, soda_trainY), axis=0)
    else:
        print("Only SODA for training!")
        first_val = int(valid_split * len(soda[0]))
        trainX, trainY = sodaX[:-first_val], soda[1][:-first_val]
        validX, validY = sodaX[-first_val:], soda[1][-first_val:]

    trainset = ENSO_Dataset(trainX, trainY)
    valset = ENSO_Dataset(validX, validY) if validX is not None else []
    testset = ENSO_Dataset(godasX, godas[1])

    if verbose:
        print('Train set:', len(trainset), 'Validation set', len(valset), 'Test set', len(testset))

    train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_training)
    test = DataLoader(testset, batch_size=batch_size, shuffle=False)
    val = None if valset == [] else DataLoader(valset, batch_size=batch_size, shuffle=False)
    del trainset, valset, testset
    return train, val, test

def reformat_cnn_data(lead_months=3, window=3, use_heat_content=False,
                      lon_min=0, lon_max=360,
                      lat_min=-55, lat_max=60,
                      data_dir="data/",
                      sample_file='CMIP5.input.36mn.1861_2001.nc',  # Input of training set
                      label_file='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
                      sst_key="sst",
                      get_valid_nodes_mask=False,
                      get_valid_coordinates=False
                      ):
    """
    :param lon_min, lon_max, lat_min, lat_max: all inclusive
    """
    import pandas as pd
    lat_p1, lat_p2 = int((lat_min + 55) / 5), min(int((lat_max + 55) / 5), 23)
    lon_p1, lon_p2 = int(lon_min / 5), min(int(lon_max / 5), 71)
    data = xa.open_dataset(f'{data_dir}/{sample_file}')
    labels = xa.open_dataset(f'{data_dir}/{label_file}')
    # Shape T' x 36 x |lat| x |lon|, want : T x 12 x |lat| x |lon|
    lat_sz = lat_p2 - lat_p1 + 1
    lon_sz = lon_p2 - lon_p1 + 1
    features = 2 if use_heat_content else 1
    feature_names = ["sst", "heat_content"] if use_heat_content else ["sst"]

    filtered_region = data.sel(
        {'lat': slice(lat_min, lat_max), 'lon': slice(lon_min, lon_max)}
    )
    filtered_region = filtered_region.rename({"lev": "window", "time": "year"})  # rename coordinate name
    X_all_target_mons = np.empty((data.sizes["time"], 12, features, window, lat_sz, lon_sz))
    Y_all_target_mons = np.empty((data.sizes["time"], 12))
    tg_mons = np.arange(0, 12)
    X_all_target_mons = xa.DataArray(X_all_target_mons, coords=[("year", data.get_index("time")),
                                                                ("tg-mon", tg_mons),
                                                                ("feature", feature_names),
                                                                ("window", np.arange(1, window + 1)),
                                                                ("lat", filtered_region.get_index("lat")),
                                                                ("lon", filtered_region.get_index("lon"))
                                                                ])
    if "CMIP5" not in label_file:
        X_all_target_mons.attrs["time"] = \
            [pd.Timestamp("1982-01-01") + pd.DateOffset(months=d_mon) for d_mon in
             range(len(data.get_index("time")) * 12)]

    X_all_target_mons.attrs["Lons"] = filtered_region.get_index('lon')
    X_all_target_mons.attrs["Lats"] = filtered_region.get_index('lat')
    for target_month in range(0, 12):
        '''
        target months are indices [25, 36)
        possible predictor months (for lead months<=24) are indices [0, 24]
        '''
        var_dict = {"ld_mn2": int(25 - lead_months + target_month) + 1,
                    "ld_mn1": int(25 - lead_months + target_month) + 1 - window}
        X_all_target_mons.loc[:, target_month, "sst", :, :, :] = \
            filtered_region.variables[sst_key][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]

        if use_heat_content:
            X_all_target_mons.loc[:, target_month, "heat_content", :, :, :] = \
                filtered_region.variables['t300'][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]

        Y_all_target_mons[:, target_month] = labels.variables['pr'][:, target_month, 0, 0]
    X_all_target_mons = X_all_target_mons.stack(time=["year", "tg-mon"])

    Y_time_flattened = Y_all_target_mons.reshape((-1,))
    X_node_flattened = X_all_target_mons.stack(cord=["lat", "lon"])
    X_time_and_node_flattened = X_node_flattened.transpose("time", "feature", "window", "cord")

    if get_valid_nodes_mask:
        sea = (np.count_nonzero(X_time_and_node_flattened[:, 0, 0, :], axis=0) > 0)
        if get_valid_coordinates:
            return X_time_and_node_flattened, Y_time_flattened, sea, X_time_and_node_flattened.get_index("cord")
        return X_time_and_node_flattened, Y_time_flattened, sea

    return X_time_and_node_flattened, Y_time_flattened


def load_cnn_data(lead_months=3, window=3, use_heat_content=False,
                  lon_min=0, lon_max=359,
                  lat_min=-55, lat_max=60,
                  data_dir="data/",
                  cmip5_data='CMIP5.input.36mn.1861_2001.nc',  # Input of CMIP5 training set
                  cmip5_label='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
                  soda_data='SODA.input.36mn.1871_1970.nc',  # Input of SODA training set
                  soda_label='SODA.label.nino34.12mn_3mv.1873_1972.nc',  # Label of training set
                  godas_data='GODAS.input.36mn.1980_2015.nc',  # Input of GODAS training set
                  godas_label='GODAS.label.12mn_3mv.1982_2017.nc',  # Label of training set
                  truncate_GODAS=True,  # whether to truncate it to the 1984-2017 period the CNN paper used
                  return_new_coordinates=False,
                  return_mask=False,
                  add_index_node=False, verbose=True, **kwargs
                  ):
    """

    :param lead_months:
    :param window:
    :param use_heat_content:
    :param lon_min:
    :param lon_max:
    :param lat_min:
    :param lat_max:
    :param data_dir:
    :param cmip5_data:
    :param cmip5_label:
    :param soda_data:
    :param soda_label:
    :param godas_data:
    :param godas_label:
    :param truncate_GODAS:
    :param return_new_coordinates:
    :param return_mask:
    :param target_months: if "all", the model will need to learn to give predictions for any target months,
                            if an int in [1, 12], it can focus on that specific target month/season,
                            where 1 translates to "JFM", ..., 12 to "DJF"
    :return:
    """
    cmip5, cmip5_Y, m1 = reformat_cnn_data(lead_months=lead_months, window=window,
                                           use_heat_content=use_heat_content,
                                           lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                           data_dir=data_dir + "CMIP5_CNN/", sst_key="sst1",
                                           sample_file=cmip5_data, label_file=cmip5_label,
                                           get_valid_nodes_mask=True, get_valid_coordinates=False)
    SODA, SODA_Y, m2 = reformat_cnn_data(lead_months=lead_months, window=window, use_heat_content=use_heat_content,
                                         lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                         data_dir=data_dir + "SODA/", sample_file=soda_data, label_file=soda_label,
                                         get_valid_nodes_mask=True, get_valid_coordinates=False)
    GODAS, GODAS_Y, m3 = reformat_cnn_data(lead_months=lead_months, window=window, use_heat_content=use_heat_content,
                                           lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                           data_dir=data_dir + "GODAS/", sample_file=godas_data, label_file=godas_label,
                                           get_valid_nodes_mask=True, get_valid_coordinates=False)
    if truncate_GODAS:
        start_1984 = 24  # 2 * 12
        GODAS, GODAS_Y, GODAS.attrs["time"] = GODAS[start_1984:], GODAS_Y[start_1984:], GODAS.attrs["time"][start_1984:]
    # DUE to variations due to resolution = 5deg., there are some inconsistencies in which nodes are terrestrial
    final_mask = np.logical_and(m1, np.logical_and(m2, m3))
    cmip5, SODA, GODAS = cmip5[:, :, :, final_mask], SODA[:, :, :, final_mask], GODAS[:, :, :, final_mask]
    if add_index_node:
        cmip5, SODA, GODAS = add_ONI_node(cmip5), add_ONI_node(SODA), add_ONI_node(GODAS)
        # cords = np.array(list(cords) + [(0, 205)])  # add coordinate for ONI
        final_mask = np.append(final_mask, True)  # add
        if verbose:
            print('MASKING OUT', np.count_nonzero(np.logical_not(final_mask)), 'nodes')
    cords = GODAS.indexes['cord']
    if return_new_coordinates:
        # cords = cords[final_mask]
        if return_mask:
            return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), cords, final_mask
        return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), cords
    if return_mask:
        return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), final_mask
    return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y)


def add_ONI_node(data_array):
    """

    :param data_array: A xarray DataArray of shape (#time-steps, #features, window, #nodes)
    :return: A xarray DataArray of shape (#time-steps, #features, window, #nodes + 1)
    """
    _, mask = get_index_mask(data_array[:, 0, 0, :], 'ONI', flattened_too=True, is_data_flattened=True)
    mask = np.array(mask)
    oni_cord_index = pd.MultiIndex.from_tuples([(0, 205)], names=['lat', 'lon'])
    ONI_NODE = data_array[:, :, :, mask].mean(dim='cord', keepdims=True).assign_coords({'cord': oni_cord_index})
    data_array = xa.concat((data_array, ONI_NODE), dim='cord')
    return data_array

