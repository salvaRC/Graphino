"""
Author: Salva Rühling Cachay
"""

import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
# from utilities.plotting import plot_time_series


def reload_all(model_dir, ens_dict, checkpoint_IDs=None, device='cuda', data_dir='./Data/'):
    from graphino.GCN.GCN_model import GCN
    from graphino.training import evaluate, get_dataloaders
    if checkpoint_IDs is None:
        checkpoint_IDs = ['last']

    if os.path.isfile(os.path.join(model_dir, 'last_model.pkl')):
        mdl_file = os.path.join(model_dir, 'last_model.pkl')
    else:
        mdl_file = os.path.join(model_dir, '50ep_model.pkl')

    try:
        model_dict = torch.load(mdl_file, map_location=device)
    except FileNotFoundError as e:
        print(e)
        return None, ens_dict

    params, net_params = model_dict['metadata']['params'], model_dict['metadata']['net_params']
    params['data_dir'] = data_dir
    print(net_params)

    (adj, static_feats, _), (_, valloader, testloader) = get_dataloaders(params, net_params)
    model = GCN(net_params, static_feat=static_feats, adj=adj, verbose=False, device=device)
    Y = None
    for ckpt in checkpoint_IDs:
        try:
            model_dict = torch.load(os.path.join(model_dir, f'{ckpt}_model.pkl'))
        except FileNotFoundError as e:
            print(e)
            continue
        try:
            model.load_state_dict(model_dict['model'])
        except RuntimeError as e:
            print(e)
            continue

        model.eval()
        model.to(device)

        _, stats = evaluate(valloader, model, device, return_preds=False)
        _, test_stats, Y, preds = evaluate(testloader, model, device, return_preds=True)
        print(model_dir.split('/')[-1], ckpt, "Val. rmse=", stats['rmse'],
              'TEST:', test_stats['rmse'], "Corrcoef, all-season-corrcoef =", test_stats['corrcoef'], test_stats['all_season_cc'])
        if stats['rmse'] > ens_dict[ckpt][-1][0]:
            print("Skip this one")
            continue
        ens_dict[ckpt][-1] = (stats['rmse'], preds)
        ens_dict[ckpt] = sorted(ens_dict[ckpt], key=lambda tup: tup[0])

    return Y, ens_dict


def ensemble_performance(out_dir, verbose=True, device="cuda", num_members=4, checkpoint_IDs=None, ID=None, data_dir='./Data/'):
    """
    :param device:
    :param num_members:
    :return: Tuple of
           i) Member predictions in descending order of preditive skill (wrt val set), i.e. the first has highest skill
           ii) Ground truth
    """
    if not verbose:
        print("This may take a while...")
    if checkpoint_IDs is None:
        checkpoint_IDs = ['last']
    topK = {name: [(10000, None) for _ in range(num_members)]
            for name in checkpoint_IDs
            }
    config_dir = out_dir
    added = False
    for fileID in os.listdir(config_dir):
        if ID is not None and ID not in fileID:
            print('Skipping this one')
            continue
        mdl_dir = os.path.join(config_dir, fileID)
        Y, topK = reload_all(mdl_dir, topK, device=device, checkpoint_IDs=checkpoint_IDs, data_dir=data_dir)
        if Y is not None:
            Ytrue = Y
        added = True

    assert added
    for name in checkpoint_IDs:
        member_preds = [mem[1] for mem in topK[name]]
        stats = ensemble(Ytrue, *member_preds)
        rmse_ens, cc_ens, cc2a = stats['rmse'], stats['corrcoef'], stats['all_season_cc']
        if verbose:
            print(f"ENSEMBLE PERFORMANCE {name} -- RMSE: {rmse_ens:.5f}, Corrcoef = {cc_ens:.5f}, All-season-CC={cc2a:.5f}")
    return member_preds, Ytrue


def rmse(y, preds):
    """
    :return:  The root-mean-squarred error (RMSE)  value
    """
    return np.sqrt(mean_squared_error(y, preds))


def evaluate_preds(Ytrue, preds, **kwargs):
    oni_corr = np.corrcoef(Ytrue, preds)[0, 1]
    try:
        rmse_val = rmse(Ytrue, preds)
    except ValueError as e:
        print(e)
        rmse_val = -1
    # r, p = pearsonr(Ytrue, preds)   # same as using np.corrcoef(y, yhat)[0, 1]
    oni_stats = {"corrcoef": oni_corr, "rmse": rmse_val}  # , "Pearson_r": r, "Pearson_p": p}

    try:
        # All season correlation skill = Mean of the corrcoefs for each target season
        # (whereas the corrcoef above computes it directly on the whole timeseries).
        predsTS = preds.reshape((-1, 12))
        YtestTT = Ytrue.reshape((-1, 12))
        all_season_cc = 0
        for target_mon in range(12):
            all_season_cc += np.corrcoef(predsTS[:, target_mon], YtestTT[:, target_mon])[0, 1]
        all_season_cc /= 12
        oni_stats['all_season_cc'] = all_season_cc
    except ValueError:
        pass

    return oni_stats


def ensemble(Ytrue, *args, return_preds=False, **kwargs):
    members = [mem for mem in args if mem is not None]
    n_members = len(members)
    assert n_members > 0, "An ensemble requires at least 1 member"
    preds = np.zeros(Ytrue.shape)
    for member in members:
        preds = preds + member
    preds = preds / n_members
    stats = evaluate_preds(Ytrue, preds, **kwargs, return_dict=True)
    if return_preds:
        return stats, preds
    return stats


# %%

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Torch uses CUDA?", torch.cuda.is_available())
    from utilities.utils import set_gpu
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", default='', type=str)
    parser.add_argument("--gpu_id", default=2, type=int)
    parser.add_argument("--horizon", default=6, type=int)  # number of lead months
    parser.add_argument("--type", default='50ep', type=str)
    parser.add_argument("--data_dir", default='./Data/', type=str)
    args = parser.parse_args()
    set_gpu(args.gpu_id)

    out = f'out/{args.horizon}lead/'
    checkpoint = [args.type]
    ens_members, Y = ensemble_performance(
        out, verbose=True, num_members=4,
        device=device,
        checkpoint_IDs=checkpoint, ID=args.ID, data_dir=args.data_dir
    )
