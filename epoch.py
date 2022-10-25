from config import args, cprint
import numpy as np
import torch
from utils.sample import UniformSample_original
import utils
from utils import timer
import models
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


def train(model, optimizer, dataset, epoch, writer=None):
    model.train()
    
    with timer(name="Sample"):
        S = UniformSample_original(dataset).to(args.device)

    all_idx_u, all_idx_i_pos, all_idx_i_neg = utils.shuffle(torch.Tensor(S[:, 0]).long(),
                                                            torch.Tensor(S[:, 1]).long(),
                                                            torch.Tensor(S[:, 2]).long())
    n_batch = len(all_idx_u) // args.tr_batch_size + 1
    loss_total = 0.

    for i, (idx_u, idx_i_pos, idx_i_neg) in enumerate(utils.minibatch(all_idx_u, all_idx_i_pos, all_idx_i_neg, batch_size=args.bpr_batch_size)):

        loss_batch = model.cal_loss(idx_u, idx_i_pos, idx_i_neg)

        loss_batch.backward()

        loss_total += loss_batch * len(idx_u)
        if args.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(idx_u) / args.bpr_batch_size) + i)

    time_info = timer.dict()
    timer.zero()
    return f"loss{loss_total / n_batch:.3f}-{time_info}"
    
    
def evaluate_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def evaluate(dataset, model, epoch, w=None, multicore=0):
    cprint('[TEST]')
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    model: model.LightGCN
    # eval mode with no dropout
    model = model.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = model.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(evaluate_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(evaluate_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results