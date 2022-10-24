import sys
import os
from os.path import join
import torch
# from enum import Enum
import multiprocessing
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='o lightGCN')
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help='the batch size for bpr loss training procedure')
    parser.add_argument('--recdim', type=int, default=64,
                        help='the embedding size of lightGCN')
    parser.add_argument('--layer', type=int, default=3,
                        help='the layer num of lightGCN')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='the learning rate')
    parser.add_argument('--decay', type=float, default=1e-4,
                        help='the weight decay for l2 normalizaton')
    parser.add_argument('--dropout', type=int, default=0,
                        help='using the dropout or not')
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help='the batch size for bpr loss training procedure')
    parser.add_argument('--a_fold', type=int, default=100,
                        help='the fold num used to split large adj matrix, like gowalla')
    parser.add_argument('--testbatch', type=int, default=100,
                        help='the batch size of users for testing')
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help='available datasets: [lastfm, gowalla, yelp2018, amazon-book]')
    parser.add_argument('--path', type=str, default='./checkpoints',
                        help='path to save weights')
    parser.add_argument('--topks', nargs='?', default='[20]',
                        help='@k test list')
    parser.add_argument('--tensorboard', type=int, default=1,
                        help='enable tensorboard')
    parser.add_argument('--comment', type=str, default='lgn')
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    return parser.parse_args()


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.getcwd()
CODE_PATH = ROOT_PATH
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
sys.path.append(join(CODE_PATH, 'sources'))


config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_modelsv = ['mf', 'lgn']
# config['batch_size'] = 4096
config['model_name'] = 'lgn'
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

GPU = torch.cuda.is_available()
device = torch.device('cuda:'+str(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
if dataset not in all_dataset:
    raise NotImplementedError(f'Haven\'t supported {dataset} yet!, try {all_dataset}')

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning)

map_color = {
    'red': 41,
    'green': 42,
    'yellow': 43,
    'blue': 44,
    'cyan': 46,
    'white': 47
}


def cprint(words: str, c: 'str > blue'):
    color = map_color[c]
    print(f'\033[0;30;{color}m{words}\033[0m')
