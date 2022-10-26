from os.path import join
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import args, cprint


class RecDataset(Dataset):
    """ Recommendation dataset """
    def __init__(self):
        self.device = args.device
        self.split = args.a_split
        self.folds = args.n_fold
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']

        self.path_data = args.path_data
        print(f'[info] Load dataset [{self.dataset}]')
        train_file = join(self.path_data, 'train.txt')
        test_file = join(self.path_data, 'test.txt')

        tr_u_unique, tr_i, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    tr_u_unique.append(uid)
                    trainUser.extend([uid] * len(items))
                    tr_i.extend(items)
                    self.n_item = max(self.n_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.tr_u_unique = np.array(tr_u_unique)
        self.trainUser = np.array(trainUser)
        self.tr_i = np.array(tr_i)

        with open(test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.n_item = max(self.n_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.n_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.graph = None
        print(f'{self.trainDataSize} interactions for training')
        print(f'{self.testDataSize} interactions for testing')
        print(f'{self.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_user / self.n_item}')

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.tr_i)),
                                      shape=(self.n_user, self.n_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

        self.length = len()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.all_seq[index], self.mask[index], self.gt[index]

    def get_data_mask(self, all_seq):
        """ Generate masked user sequences"""
        lens_all_seq = [len(seq) for seq in all_seq]
        len_max = max(lens_all_seq)
        all_seq_masked = [seq + [self.index_mask] * (len_max - len_seq) for seq, len_seq in zip(all_seq, lens_all_seq)]
        mask_all_seq = [[1] * len_seq + [self.index_mask] * (len_max - len_seq) for len_seq in lens_all_seq]
        return np.asarray(all_seq_masked), np.asarray(mask_all_seq), len_max


class LastFMDataset(Dataset):
    """ Recommendation dataset """
    def __init__(self):
        self.path_data = args.path_data
        self.index_mask = 0
        self.all_seq, self.mask, self.len_max = self.get_data_mask(data[0])
        self.gt = np.asarray(data[1])
        self.length = len(self.all_seq)
        self.shuffle = shuffle
        self.graph = graph

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.all_seq[index], self.mask[index], self.gt[index]

    def get_data_mask(self, all_seq):
        """ Generate masked user sequences"""
        lens_all_seq = [len(seq) for seq in all_seq]
        len_max = max(lens_all_seq)
        all_seq_masked = [seq + [self.index_mask] * (len_max - len_seq) for seq, len_seq in zip(all_seq, lens_all_seq)]
        mask_all_seq = [[1] * len_seq + [self.index_mask] * (len_max - len_seq) for len_seq in lens_all_seq]
        return np.asarray(all_seq_masked), np.asarray(mask_all_seq), len_max


def collate_fn(seq_batch, mask_batch, gt_batch):
    """ Collate function, as required by PyTorch. """
    max_num_item_seq = np.max([len(np.unique(seq)) for seq in seq_batch])
    seq_alias_batch, items_batch, A_batch = [], [], []
    for seq in seq_batch:
        items_seq = np.unique(seq)
        items_batch.append(items_seq.tolist() + (max_num_item_seq - len(items_seq)) * [0])
        A_seq = np.zeros((max_num_item_seq, max_num_item_seq))  # Adjacency matrix for sequential
        for i in np.arange(len(seq) - 1):
            # For edges, seq[i] is in, and seq[i+1] is out
            if seq[i + 1] == 0:
                break
            in_index = np.where(items_seq == seq[i])[0][0]
            out_index = np.where(items_seq == seq[i + 1])[0][0]
            A_seq[in_index][out_index] = 1

        A_seq_in_sum = np.sum(A_seq, 0)
        A_seq_in_sum[np.where(A_seq_in_sum == 0)] = 1  # Add 1 for all nodes with 0 indegree
        A_seq_in = np.divide(A_seq, A_seq_in_sum)

        A_seq_out_sum = np.sum(A_seq, 1)
        A_seq_out_sum[np.where(A_seq_out_sum == 0)] = 1
        A_seq_out = np.divide(A_seq.transpose(), A_seq_out_sum)

        A_seq = np.concatenate([A_seq_in, A_seq_out]).transpose()
        A_batch.append(A_seq)
        seq_alias_batch.append([np.where(items_seq == i)[0][0] for i in seq])

    A_batch = torch.FloatTensor(np.array(A_batch))
    items_batch = torch.LongTensor(np.array(items_batch))
    seq_alias_batch = torch.LongTensor(np.array(seq_alias_batch))
    mask_batch = torch.LongTensor(np.array(mask_batch))
    gt_batch = torch.LongTensor(gt_batch)

    return A_batch, items_batch, seq_alias_batch, mask_batch, gt_batch


def get_dataloader():
    if args.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        return RecDataset()
    else:
        assert args.dataset == 'lastfm'
        return LastFMDataset()
