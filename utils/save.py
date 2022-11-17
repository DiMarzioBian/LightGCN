from os.path import join
import re
import numpy as np
import torch


def load_ckpt(args, model, scheduler):
    device_old = re.split('-', args.path_ckpt)[2]
    ckpt = torch.load(join(args.path_ckpt, args.load), map_location={f'cuda:{device_old}': f'cuda:{args.device}'})
    model = model.load_state_dict(ckpt['params_model'])
    scheduler = scheduler.load_state_dict(ckpt['params_scheduler'])
    return ckpt['epoch_cur'], model, scheduler


def save_ckpt(args, epoch_cur, model, scheduler):
    ckpt = {
        'epoch_cur': epoch_cur,
        'params_model': model.state_dict(),
        'params_scheduler': scheduler.state_dict()
    }
    torch.save(ckpt, join(args.path_ckpt, f'{args.model}-{args.dataset}-{args.device}-{args.n_layers}-{args.d_latent}'
                                          f'.pth.tar'))


def minibatch(args, *tensors, **kwargs):
    batch_size = kwargs.get('batch_size', args.tr_batch_size)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
