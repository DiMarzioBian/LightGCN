import os
import time
import numpy as np
from scipy import sparse as sp
import pandas as pd
import argparse

from utils import MAPPING_DATASET


def read_amazon(dataset: str, path: str):
    df = pd.read_json(path, lines=True)
    df = df.loc[:, ['reviewerID', 'asin', 'unixReviewTime']]
    df.columns = ['user', 'item', 'ts']
    df.sort_values(by=['ts', 'user'], inplace=True)
    print(f'\n[Info] Successfully loaded dataset {dataset}.')
    return df


def read_gowalla(dataset: str, path: str):
    df = pd.read_json(path, lines=True)
    df = df.loc[:, ['reviewerID', 'asin', 'unixReviewTime']]
    df.columns = ['user', 'item', 'ts']
    df.sort_values(by=['ts', 'user'], inplace=True)
    print(f'\n[Info] Successfully loaded dataset {dataset}.')
    return df


def read_yelp2018(dataset: str, path: str):
    df = pd.read_json(path, lines=True)
    df = df.loc[:, ['reviewerID', 'asin', 'unixReviewTime']]
    df.columns = ['user', 'item', 'ts']
    df.sort_values(by=['ts', 'user'], inplace=True)
    print(f'\n[Info] Successfully loaded dataset {dataset}.')
    return df


def read_lastfm(dataset: str, path: str):
    if dataset == 'ml-100k':
        df = pd.read_csv(path, header=None, sep='\t', usecols=[0, 1, 3])
    else:
        df = pd.read_csv(path, header=None, sep='::', usecols=[0, 1, 3], engine='python')
    df.columns = ['user', 'item', 'ts']
    df.sort_values(by=['ts', 'user'], inplace=True)
    print(f'\n[Info] Successfully loaded dataset "{dataset}".')
    return df


def filter_cold_start(df: pd.DataFrame):
    """ filter out cold-start item and users appears less than 5 """
    u_5 = df['user'].value_counts()[df['user'].value_counts() >= 5].index
    i_5 = df['item'].value_counts()[df['item'].value_counts() >= 5].index

    if len(df['user'].unique()) == len(u_5) and len(df['item'].unique()) == len(i_5):
        return df
    else:
        df = df[df['user'].isin(u_5)]
        df = df[df['item'].isin(i_5)]
        return filter_cold_start(df)


def reindex_data(df: pd.DataFrame, ts_unit: int, to_float=False):
    """ Re-index both users and items """
    map_u, map_v = {}, {}
    list_u, list_v = df['user'].unique().tolist(), df['item'].unique().tolist()

    for i, idx_u in enumerate(list_u):
        map_u[idx_u] = i
    for i, idx_v in enumerate(list_v):
        map_v[idx_v] = i

    df['user'] = [map_u[u] for u in df['user'].tolist()]
    df['item'] = [map_v[v] for v in df['item'].tolist()]
    if to_float:
        df['ts'] = ((df['ts'] - df['ts'].min()) / ts_unit).astype(float)
    else:
        df['ts'] = ((df['ts'] - df['ts'].min()) / ts_unit).astype(int)
    print('\n[info] Dataset contains', df.shape[0], 'interactions,', len(list_u), 'users and', len(list_v), 'items.')

    return df, len(list_u), len(list_v)


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print('\n[info] CSV save to file:', path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mlm')
    args = parser.parse_args()

    # detect dataset
    (dataset, f_name) = MAPPING_DATASET[args.dataset]
    path_raw = os.getcwd() + f'/data/{f_name}'
    path_csv = os.getcwd() + f'/data_processed/{dataset}_5.csv'

    if os.path.exists(path_raw):
        try:
            ts_unit = MAPPING_TS_UNIT[dataset]
            print(f'\n[Info] Successfully detect dataset "{dataset}" and set unit timestamp.')
        except KeyError:
            raise KeyError(f'Please set timestamp unit for dataset "{dataset}" in constant.py file.')
    else:
        raise FileNotFoundError(f'Raw dataset "{dataset}" not found in root path.')

    # read and reindex dataset
    if dataset in ['ml-100k', 'ml-1m']:
        df_raw = read_movielens(dataset, path_raw)
        df_5_raw = filter_cold_start(df_raw)
    elif dataset == 'yoochoosebuy':
        df_raw = read_yoochoose(dataset, path_raw)
        df_5_raw = filter_cold_start(df_raw)
    else:
        df_5_raw = read_amazon(dataset, path_raw)

    df_5, n_user, n_item = reindex_data(df_5_raw, ts_unit, to_float=dataset in ['ml-100k', 'ml-1m', 'yoochoosebuy_cope'])
    save_csv(df_5, path_csv)


if __name__ == '__main__':
    main()
