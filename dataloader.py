from os.path import join
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from config import args, cprint


class LastFM(Dataset):
    def __init__(self, path='./data/lastfm'):
        # train or test
        cprint('loading [last fm]')
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']
        self.device = args.device

        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        testData = pd.read_table(join(path, 'test1.txt'), header=None)
        trustNet = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()

        trustNet -= 1
        trainData -= 1
        testData -= 1
        self.n_user = args.n_user = 1892
        self.n_item = args.n_item = 4489
        self.trustNet = trustNet
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])
        self.graph = None
        print(f'LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_user/self.n_item}')
        
        # (users,users)
        self.socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_user,self.n_user))
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_user,self.n_item))
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.allNeg = []
        allItems = set(range(self.n_item))
        for i in range(self.n_user):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def get_sparse_graph(self):
        if self.graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_user])
            second_sub = torch.stack([item_dim+self.n_user, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_user+self.n_item, self.n_user+self.n_item]))
            dense = self.graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_user+self.n_item, self.n_user+self.n_item]))
            self.graph = self.graph.coalesce().to(self.device)
        return self.graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)


class Loader(Dataset):
    def __init__(self, path='./data/gowalla'):
        # train or test
        cprint(f'loading [{path}]')
        self.device = args.device
        self.split = args.a_split
        self.folds = args.n_fold
        self.mode_dict = {'train': 0, 'val':1, 'test': 2}
        self.mode = self.mode_dict['train']
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.len_tr = 0
        self.len_val = 0
        self.len_te = 0

        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.n_item = max(self.n_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

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
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.n_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f'{args.dataset} is ready to go')
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_user + self.n_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_user + self.n_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self.sp_mat_to_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def sp_mat_to_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def get_sparse_graph(self):
        print('loading adjacency matrix')
        if self.graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print('successfully loaded...')
                norm_adj = pre_adj_mat
            except :
                print('generating adjacency matrix')
                s = time()
                adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f'costing {end-s}s, saved norm_mat...')
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split:
                self.graph = self._split_A_hat(norm_adj)
                print('done split matrix')
            else:
                self.graph = self.sp_mat_to_tensor(norm_adj)
                self.graph = self.graph.coalesce().to(self.device)
                print('don\'t split the matrix')
        return self.graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems


def get_dataloader():
    if args.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        return Loader(path='./data/' + args.dataset)
    else:
        assert args.dataset == 'lastfm'
        return LastFM(path='./data/' + args.dataset)
