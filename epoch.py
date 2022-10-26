import time
import torch

from config import args, cprint
from metrics import *
from sample import UniformSample_original
from utils import minibatch, shuffle
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


def train(model, optimizer, dataset, epoch, writer=None):
    S = UniformSample_original(dataset).to(args.device)
    all_idx_u, all_idx_i_pos, all_idx_i_neg = shuffle(torch.Tensor(S[:, 0]).long(),
                                                            torch.Tensor(S[:, 1]).long(),
                                                            torch.Tensor(S[:, 2]).long())
    loss_total = 0.

    model.train()
    optimizer.zero_grad()

    time_start = time.time()
    for i, (idx_u, idx_i_pos, idx_i_neg) in enumerate(minibatch(all_idx_u, all_idx_i_pos, all_idx_i_neg, batch_size=args.bpr_batch_size)):
        loss_batch = model.cal_loss(idx_u, idx_i_pos, idx_i_neg)
        loss_batch.backward()
        optimizer.step()

        loss_total += loss_batch * len(idx_u)
        if args.tensorboard:
            writer.add_scalar(f'BPRLoss/BPR', loss_batch, epoch * int(len(idx_u) / args.bpr_batch_size) + i)

    return loss_total / len(all_idx_u), time.time() - time_start


def evaluate_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = get_label(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in args.topk:
        ret = cal_recall(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(cal_ndcg(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}
        
            
def evaluate(dataset, model, epoch, w=None, multicore=0):
    cprint('[TEST]')
    testDict: dict = dataset.testDict
    model: model.LightGCN
    # eval mode with no dropout
    model = model.eval()
    max_K = max(args.topk)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    res = {'loss': 0,
           'recall': np.zeros(len(args.topk)),
           'ndcg': np.zeros(len(args.topk))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert args.eval_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        auc_record = []
        total_batch = len(users) // args.eval_batch_size + 1
        for batch_users in minibatch(users, batch_size=args.eval_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args.device)

            rating = model.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            auc_record.extend([cal_auc(rating[i], dataset, test_data) for i, test_data in enumerate(groundTrue)])
            del rating

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)

        pre_res = []
        for x in X:
            pre_res.append(evaluate_one_batch(x))
        scale = float(args.eval_batch_size/len(users))

        for r in pre_res:
            res['recall'] += r['recall']
            res['ndcg'] += r['ndcg']
        res['recall'] /= float(len(users))
        res['ndcg'] /= float(len(users))
        res['auc'] = np.mean(auc_record)
        if args.tensorboard:
            w.add_scalars(f'Test/Recall@{args.topk}',
                          {str(args.topk[i]): res['recall'][i] for i in range(len(args.topk))}, epoch)
            w.add_scalars(f'Test/NDCG@{args.topk}',
                          {str(args.topk[i]): res['ndcg'][i] for i in range(len(args.topk))}, epoch)
            w.add_scalars(f'Test/AUC@{args.topk}',
                          {str(args.topk[i]): res['auc'][i] for i in range(len(args.topk))}, epoch)
        print(res)
        return res
