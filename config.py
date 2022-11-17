import os
from os.path import join
import torch
import argparse

IDX_PAD = 0

MAPPING_COLOR = {
    'red': 41,
    'green': 42,
    'yellow': 43,
    'blue': 44,
    'cyan': 46,
    'white': 47
}

MAPPING_DATASET = {
    'book': ['amazon-book', 'Movies_and_TV_5.json'],
    'garden': ['amazon-garden', 'Patio_Lawn_and_Garden_5.json'],
    'gowalla': ['gowalla', 'Patio_Lawn_and_Garden_5.json'],
    'yelp': ['yelp2018', 'Patio_Lawn_and_Garden_5.json'],
    'lastfm': ['lastfm', 'Patio_Lawn_and_Garden_5.json'],
}


def parse_args():
    parser = argparse.ArgumentParser(description='lightGCN')

    # experiment settings
    parser.add_argument('--a_split', action='store_true', help='')
    parser.add_argument('--bigdata', action='store_true', help='')

    # model settings
    parser.add_argument('--model', type=str, default='lightgcn', help='lightgcn, mf')
    parser.add_argument('--d_latent', type=int, default=64, help='latent size of embedding')
    parser.add_argument('--n_layer', type=int, default=3, help='the layer num of lightGCN')
    parser.add_argument('--pretrain', action='store_true', help='whether we use pretrained weight or not')
    parser.add_argument('--init_normal', action='store_false', help='use normal initialization or uniform')
    parser.add_argument('--dropout', type=float, default=0, help='dropout ratio')

    # general settings
    parser.add_argument('--dataset', type=str, default='gowalla', help='lastfm, gowalla, yelp, book, garden')
    parser.add_argument('--tensorboard', action='store_false', help='enable tensorboard')
    parser.add_argument('--cuda', type=int, default=0, help='index of cuda used')
    parser.add_argument('--load', type=str, default='', help='name of model params to be loaded')
    parser.add_argument('--topk', nargs='?', default='[20]', help='@k test list')
    parser.add_argument('--negk', type=int, default=1, help='# negative items')
    parser.add_argument('--comment', type=str, default='lgn')

    # training settings
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='num of data loader')
    parser.add_argument('--tr_batch_size', type=int, default=2048, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=100, help='evaluating batch size')
    parser.add_argument('--n_fold', type=int, default=100,
                        help='the fold num used to split large adj matrix, like gowalla')

    # optimizer settings
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--lr_patience', type=int, default=50)
    parser.add_argument('--lr_factor', type=float, default=0.5, help='gamma value')
    parser.add_argument('--lr_n_decay', type=int, default=100)

    # set up
    args = parser.parse_args()
    args.idx_pad = IDX_PAD

    (args.dataset, _) = MAPPING_DATASET[args.dataset]
    args.path_root = os.getcwd()
    args.path_data = join(join(args.path_root, 'data'), args.dataset)
    args.path_board = join(args.path_root, 'runs')
    args.path_ckpt = join(args.path_root, 'checkpoints')

    if torch.cuda.is_available() and args.cuda in torch.get_all_devices():
        args.device = torch.device('cuda:' + str(args.cuda))
    else:
        raise NotImplementedError('CPU or selected GPU is not available.')

    return parser.parse_args()


def cprint(words: str, c: str = 'blue'):
    color = MAPPING_COLOR[c]
    print(f'\033[0;30;{color}m{words}\033[0m')


