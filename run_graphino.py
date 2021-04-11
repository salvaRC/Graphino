"""
Author: Salva Rühling Cachay
"""

import argparse
import json
import os
import time
import warnings

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Training settings
from eval_gcn import ensemble_performance
from graphino.training import evaluate, train_epoch, get_dataloaders
from graphino.GCN.GCN_model import GCN

from utilities.utils import set_gpu, set_seed
from utilities.model_logging import update_tqdm, log_epoch_vals, save_model
from utilities.optimization import get_optimizer, get_loss

if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # or use: "once"
    parser = argparse.ArgumentParser(description='PyTorch ENSO Time series forecasting')
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--horizon", default=3, type=int)
    parser.add_argument("--out", default='out', type=str)
    parser.add_argument("--optim", default='adam', type=str)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", default=50, type=int)  #
    parser.add_argument("--data_dir", default='Data/', type=str)
    parser.add_argument("--grid_edges", default='false', type=str)
    args = parser.parse_args()
    if args.gpu_id >= 0:
        device = 'cuda'
        set_gpu(args.gpu_id)
    else:
        device = 'cpu'
    base_dir = f'{args.out}/{args.horizon}lead/'
    adj = None
    if args.grid_edges.lower() == 'true':
        base_dir += 'GRID_EDGES_'

    config_files = ['250x100', '250x250', '200x200x200_Mean+Sum', '250x250x250_Mean+Sum']
    ID = str(time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'))
    for i, config_file in enumerate(config_files):
        with open(f'configs/GCN_{config_file}.json', 'r') as f:
            config = json.load(f)
        params, net_params = config['params'], config['net_params']
        params['horizon'] = args.horizon
        params['data_dir'] = args.data_dir + '/'
        params['optimizer'] = args.optim
        params['weight_decay'] = args.weight_decay or params['weight_decay']
        params['lr'] = args.lr or params['lr']
        params['epochs'] = args.epochs or params['epochs']
        params['grid_edges'] = True if args.grid_edges.lower() == 'true' else False
        set_seed(params['seed'])

        (adj, static_feats, _), (trainloader, valloader, testloader) = get_dataloaders(params, net_params)
        ckpt_dir = base_dir + config_file + '_' + ID + '/'
        writer = SummaryWriter(log_dir=ckpt_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # Model and optimizer
        model = GCN(net_params, static_feat=static_feats, adj=adj)
        optimizer = get_optimizer(params['optimizer'], model, lr=params['lr'],
                                  weight_decay=params['weight_decay'], nesterov=params['nesterov'])
        criterion = get_loss(params['loss'])

        # Train model
        t_total = time.time()
        model = model.to(device)
        val_stats = None
        best_val_loss = cur_val = 1000
        print('Params', params)
        print('Net params', net_params)
        with tqdm(range(1, params['epochs'] + 1)) as t:
            for epoch in t:
                t.set_description(f'Graphino-{args.horizon}h-{config_file}')
                start_t = time.time()
                loss, num_edges = train_epoch(trainloader, model, criterion, optimizer, device, epoch)
                duration = time.time() - start_t

                if valloader is not None:
                    # Note that the default 'validation set' is included in the training set (=SODA),
                    # and is not used at all.
                    _, val_stats = evaluate(valloader, model, device=device)
                _, test_stats = evaluate(testloader, model, device=device)

                log_epoch_vals(writer, loss, epoch, val_stats=val_stats, test_stats=test_stats)  # tensorboard logging
                update_tqdm(t, loss, n_edges=num_edges, time=duration, val_stats=val_stats, test_stats=test_stats)

        save_model(model, ckpt_dir, params, net_params, optimizer, epoch, ID='last_model.pkl')
        writer.flush()
        writer.close()
    ensemble_performance(base_dir, verbose=True, num_members=4, ID=ID)
