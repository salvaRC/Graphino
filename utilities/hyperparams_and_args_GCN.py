"""
Author: Salva Rühling Cachay
"""

import argparse
import os
from utilities.utils import set_gpu


def get_argparser():
    parser = argparse.ArgumentParser(description='PyTorch ENSO forecasting')
    parser.add_argument("--data_dir", type=str, default='Data/')
    parser.add_argument('--gpu_id', default=0, help="Please give a value for gpu id")
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--model_dir', type=str, default='./out/', help='')
    parser.add_argument('--expID', type=str, default='', help='')
    parser.add_argument('--window', type=int, default=3,
                        help='input sequence length')  # how many time steps used for prediction?...
    parser.add_argument('--horizon', type=int, default=6)  # predict horizon months in advance...

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay rate')

    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--shuffle', default='True')  # shuffle training batches?
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument("--resolution", type=int, default=5, help="Which grid resolution to use")

    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--dataset', default="Ham", help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', default=41, type=int, help="Please give a value for seed")
    parser.add_argument('--optimizer', default='SGD', help="Please give a value for init_lr")
    parser.add_argument('--lr', default=0.005, type=float, help="Please give a value for learning rate")
    parser.add_argument('--lr_reduce_factor', type=float, help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', type=float, help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', type=float, help="Please give a value for min_lr")
    parser.add_argument('--nesterov', default='True', type=str, help="Please give a value for learning rate")
    parser.add_argument('--L', default=2, type=int, help="Please give a value for L")
    parser.add_argument('--hidden_dim', default=250, type=int, help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', default=100, type=int, help="Please give a value for out_dim")
    parser.add_argument('--residual', default='True', help="Please give a value for residual")
    parser.add_argument('--readout', default='mean', type=str, help="Please give a value for readout")
    parser.add_argument('--batch_norm', default='True', help="Please give a value for batch_norm")
    parser.add_argument('--mlp_batch_norm', default='True', help="Please give a value for batch_norm")
    parser.add_argument('--out_func', default='identity', help="Please give a value for batch_norm")

    parser.add_argument('--self_loop', default='True', help="Please give a value for self_loop")
    parser.add_argument('--scheduler', default="No", help="Please give a value for using a schedule")
    parser.add_argument('--loss', default="MSE", help="Please give a value for ")
    parser.add_argument('--act', default="ELU", help="Please give a value for ")
    parser.add_argument('--tanh_alpha', default=0.1, type=float)
    parser.add_argument('--sig_alpha', default=2.0, type=float)
    parser.add_argument('--adj_dim', default=50, type=int)
    parser.add_argument('--avg_edges_per_node', default=8, type=float)
    parser.add_argument('--index_node', default='True')
    parser.add_argument('--jumping_knowledge', default='True', type=str)
    ########

    parser.add_argument("--use_heat_content", default='True',
                        help="Whether to use heat content anomalies")
    parser.add_argument("--useCMIP5", default='True',
                        help="Whether too concatenate cmip5 with soda for fine tuning, or only use soda")
    parser.add_argument('--lon_min', type=int, default=0, help='Longitude min. (Eastern)')
    parser.add_argument('--lon_max', type=int, default=360, help='Longitude max. (Eastern)')
    parser.add_argument('--lat_min', type=int, default=-55, help='Latitude min. (Southern)')
    parser.add_argument('--lat_max', type=int, default=60, help='Latitude max. (Southern)')
    parser.add_argument("--validation_set", type=str, default='SODA', help='CMIP5 or SODA')
    parser.add_argument('--validation_frac', type=float, default=0, help='Validation set fraction')
    args = parser.parse_args()
    # parameters
    params = dict()
    params['model_dir'] = args.model_dir
    params['window'], params['horizon'] = args.window, args.horizon
    params['lon_min'], params['lon_max'] = args.lon_min, args.lon_max
    params['lat_min'], params['lat_max'] = args.lat_min, args.lat_max
    params['data_dir'] = args.data_dir
    params["validation_set"], params["validation_frac"] = args.validation_set, args.validation_frac
    params['useCMIP5'] = True if args.useCMIP5.lower() == 'true' else False
    params['use_heat_content'] = True if args.use_heat_content.lower() == 'true' else False
    params['nesterov'] = True if args.nesterov.lower() == 'true' else False
    params['loss'] = args.loss
    params['seed'] = int(args.seed)
    params['shuffle'] = True if args.shuffle.lower() == 'true' else False
    params['epochs'] = int(args.epochs)
    params['batch_size'] = int(args.batch_size)
    params['lr'] = float(args.lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    params['weight_decay'] = float(args.weight_decay)
    params['optimizer'] = args.optimizer or 'Adam'
    params["horizon"] = args.horizon
    params["window"] = args.window
    params['scheduler'] = args.scheduler

    # network parameters
    net_params = dict()
    net_params['activation'] = args.act
    net_params['in_dim'] = 2 * params['window'] if params['use_heat_content'] else params['window']
    net_params['self_loop'] = args.self_loop
    net_params['tanh_alpha'] = args.tanh_alpha
    net_params['sig_alpha'] = args.sig_alpha
    net_params['adj_dim'] = args.adj_dim
    net_params['out_func'] = args.out_func
    net_params['avg_edges_per_node'] = args.avg_edges_per_node
    net_params['jumping_knowledge'] = True if args.jumping_knowledge == 'True' else False
    net_params['index_node'] = True if args.index_node == 'True' else False
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual in ['True', True] else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    net_params['batch_norm'] = True if args.batch_norm.lower() == 'true' else False
    net_params['mlp_batch_norm'] = True if args.mlp_batch_norm.lower() == 'true' else False

    def prefix():
        s = f"{args.horizon}h_"
        if not params['useCMIP5']:
            s += "onlySODA_"
        if params['use_heat_content']:
            s += "+HC_"
        s += f"{params['window']}w_{net_params['L']}L_"
        if args.dropout > 0:
            s += f"{args.dropout}dout_"
        s += f"{args.readout.upper()}_"
        s += net_params['activation'] + 'act_'
        s += args.optimizer + 'Optim_'
        s += args.scheduler + 'Sched_'
        if net_params['self_loop']:
            s += f"SelfLoops_"
        s += f"{params['batch_size']}bs_"
        s += f"{params['lr']}lr_"
        if args.weight_decay > 0:
            s += f"{args.weight_decay}wd_"
        s += f"{net_params['hidden_dim']}h&{net_params['out_dim']}oDim"
        s += f'{args.tanh_alpha}t&{args.sig_alpha}sAlpha_'
        if not params['shuffle']:
            s += 'noShuffle_'
        s += f'{args.seed}seed_'

        s += args.expID
        return s

    set_gpu(args.gpu_id)
    params['ID'] = prefix()
    return params, net_params

