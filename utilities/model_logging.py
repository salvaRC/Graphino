"""
Author: Salva Rühling Cachay
"""

from datetime import datetime
import torch


def log_epoch_vals(writer, loss, epoch, val_stats=None, test_stats=None):
    if writer is None:
        return
    writer.add_scalar('train/_loss', loss, epoch)
    # writer.add_scalar('train/_mae', stats['mae'], epoch)
    if val_stats is not None:
        writer.add_scalar('val/_mae', val_stats['mae'], epoch)
        writer.add_scalar('val/_rmse', val_stats['rmse'], epoch)
        writer.add_scalar('val/_corrcoef', val_stats['corrcoef'], epoch)
        writer.add_scalar('val/_all_season_cc', val_stats['all_season_cc'], epoch)
    writer.add_scalar('test/_mae', test_stats['mae'], epoch)
    writer.add_scalar('test/_rmse', test_stats['rmse'], epoch)
    writer.add_scalar('test/_corrcoef', test_stats['corrcoef'], epoch)
    writer.add_scalar('test/_all_season_cc', test_stats['all_season_cc'], epoch)


def set_if_exists(dictio_from, dictio_to, key, prefix):
    if key in dictio_from:
        dictio_to[f'{prefix}_{key}'.lstrip('_')] = dictio_from[key]


def update_tqdm(tq, train_loss, val_stats=None, test_stats=None, **kwargs):
    def get_stat_dict(dictio, prefix, all=False):
        dict_two = dict()
        set_if_exists(dictio, dict_two, 'rmse', prefix)
        set_if_exists(dictio, dict_two, 'corrcoef', prefix)
        set_if_exists(dictio, dict_two, 'all_season_cc', prefix)

        if all:
            set_if_exists(dictio, dict_two, 'mae', prefix)
        return dict_two

    if val_stats is None:
        if test_stats is None:
            tq.set_postfix(train_loss=train_loss, **kwargs)
        else:
            test_print = get_stat_dict(test_stats, 'test')
            tq.set_postfix(train_loss=train_loss, **test_print, **kwargs)
    else:
        val_print = get_stat_dict(val_stats, 'val', all=True)
        if test_stats is None:
            tq.set_postfix(train_loss=train_loss, **val_print, **kwargs)
        else:
            test_print = get_stat_dict(test_stats, 'test')
            tq.set_postfix(train_loss=train_loss, **val_print, **test_print, **kwargs)


def save_model(model, model_dir, params, net_params, optimizer, epoch, ID='model.pkl'):
    checkpoint_dict = {
        'model': model.state_dict(),
        'epoch': epoch,
        'name': str(model),
        'optimizer': optimizer.state_dict(),
        'metadata': {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'params': params,
            'net_params': net_params
        }
    }
    # In case a model dir was given --> save best model (wrt validation data)
    if model_dir is not None:
        torch.save(checkpoint_dict, f'{model_dir}/{ID}')

