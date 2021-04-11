"""
Author: Salva Rühling Cachay
"""

import os
import torch
from graphino.GCN.GCN_model import GCN
from graphino.training import evaluate, get_dataloaders
# from utilities.plotting import plot_time_series
from evaluation.eval import ensemble

def reload_all(model_dir, ens_dict, checkpoint_IDs=None, device='cuda'):
    if checkpoint_IDs is None:
        checkpoint_IDs = ['last']
    try:
        model_dict = torch.load(os.path.join(model_dir, 'last_model.pkl'))
    except FileNotFoundError as e:
        print(e)
        return None, ens_dict
    params, net_params = model_dict['metadata']['params'], model_dict['metadata']['net_params']
    print(net_params)
    (adj, static_feats, _), (_, valloader, testloader) = get_dataloaders(params, net_params)
    model = GCN(net_params, static_feat=static_feats, adj=adj, verbose=False)
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
        print(model_dir.split('/')[-1], ckpt,
              stats['rmse'], 'TEST:', test_stats['rmse'], test_stats['corrcoef'], test_stats['all_season_cc'])
        if stats['rmse'] > ens_dict[ckpt][-1][0]:
            print("Skip this one")
            continue
        ens_dict[ckpt][-1] = (stats['rmse'], preds)
        ens_dict[ckpt] = sorted(ens_dict[ckpt], key=lambda tup: tup[0])

    return Y, ens_dict


def ensemble_performance(out_dir, verbose=True, device="cuda", num_members=4, ID=None):
    """
    :param device:
    :param num_members:
    :return: Tuple of
           i) Member predictions in descending order of preditive skill (wrt val set), i.e. the first has highest skill
           ii) Ground truth
    """
    if not verbose:
        print("This may take a while...")
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
        Y, topK = reload_all(mdl_dir, topK, device=device, checkpoint_IDs=checkpoint_IDs)
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


# %%

if __name__ == '__main__':
    import argparse
    from utilities.utils import set_gpu
    parser = argparse.ArgumentParser(description='PyTorch ENSO Time series forecasting')
    parser.add_argument("--gpu_id", default=2, type=int)
    parser.add_argument("--horizon", default=6, type=int)
    parser.add_argument("--type", default='last', type=str)
    parser.add_argument("--ID", default='', type=str)
    args = parser.parse_args()
    set_gpu(args.gpu_id)
    out = f'out/{args.horizon}lead/'
    # %%
    ens_members, Y = ensemble_performance(out, verbose=True, num_members=4, ID=args.ID)
