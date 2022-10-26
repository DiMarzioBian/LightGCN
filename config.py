import os
from os.path import join
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='lightGCN')

    # model settings
    parser.add_argument('--model', type=str, default='lightgcn', help='lightgcn, mf')
    parser.add_argument('--d_latent', type=int, default=64, help='latent size of embedding')
    parser.add_argument('--n_layer', type=int, default=3, help='the layer num of lightGCN')
    parser.add_argument('--pretrain', action='store_true', help='whether we use pretrained weight or not')
    parser.add_argument('--init_normal', action='store_false', help='use normal initialization or uniform')
    parser.add_argument('--dropout', type=int, default=0, help='using the dropout or not')
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help='the batch size for bpr loss training procedure')

    # general settings
    parser.add_argument('--dataset', type=str, default='gowalla', help='lastfm, gowalla, yelp2018, amazon-book')
    parser.add_argument('--tensorboard', action='store_false', help='enable tensorboard')
    parser.add_argument('--device', type=int, default=0, help='index of avail cuda')
    parser.add_argument('--load', type=str, default='', help='name of model params to be loaded')
    parser.add_argument('--topk', nargs='?', default='[20]', help='@k test list')
    parser.add_argument('--comment', type=str, default='lgn')

    # training settings
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--tr_batch_size', type=int, default=2048, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=100, help='evaluating batch size')
    parser.add_argument('--n_fold', type=int, default=100,
                        help='the fold num used to split large adj matrix, like gowalla')

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--lr_patience', type=int, default=50)
    parser.add_argument('--lr_factor', type=float, default=0.5, help='i.e. gamma value')
    parser.add_argument('--lr_n_decay', type=int, default=100)

    return parser.parse_args()


map_color = {
    'red': 41,
    'green': 42,
    'yellow': 43,
    'blue': 44,
    'cyan': 46,
    'white': 47
}


def cprint(words: str, c: str = 'blue'):
    color = map_color[c]
    print(f'\033[0;30;{color}m{words}\033[0m')


args = parse_args()

args.path_root = os.getcwd()
args.path_data = join(args.path_root, 'data')
args.path_board = join(args.path_root, 'runs')
args.path_ckpt = join(args.path_root, 'checkpoints')

args.a_split = False
args.bigdata = False

if torch.cuda.is_available():
    args.cuda = torch.device('cuda:' + str(args.device))
else:
    raise NotImplementedError('Model does not support CPU.')
