import os
import pandas as pd
import argparse

from arguments import MAPPING_DATASET


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


def filter_cold_start(df: pd.DataFrame, k_cold_start: int):
    """ filter out cold-start item and users appears less than [k_cold_start] """
    u_5 = df['user'].value_counts()[df['user'].value_counts() >= k_cold_start].index
    i_5 = df['item'].value_counts()[df['item'].value_counts() >= k_cold_start].index

    if len(df['user'].unique()) == len(u_5) and len(df['item'].unique()) == len(i_5):
        return df
    else:
        df = df[df['user'].isin(u_5)]
        df = df[df['item'].isin(i_5)]
        return filter_cold_start(df)


def reindex_data(df: pd.DataFrame):
    dict_u, dict_i = {}, {}
    list_u, list_i = df['user'].unique().tolist(), df['item'].unique().tolist()

    for ii, idx_u in enumerate(list_u):
        dict_u[idx_u] = ii
    for ii, idx_i in enumerate(list_i):
        dict_i[idx_i] = ii

    df['user'] = [dict_u[u] for u in df['user'].tolist()]
    df['item'] = [dict_i[ii] for ii in df['item'].tolist()]
    print('\n[info] Dataset contains', df.shape[0], 'interactions,', len(list_u), 'users and', len(list_i), 'items.')

    return df


def save_data(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print('\n[info] CSV save to file:', path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mlm')
    parser.add_argument('--k_cold_start', type=int, default=5)
    args = parser.parse_args()

    # detect dataset
    (dataset, f_name) = MAPPING_DATASET[args.dataset]
    path_raw = os.path.dirname(os.getcwd()) + f'/raw/{f_name}'
    path_data = os.path.dirname(os.getcwd()) + f'/data/{dataset}/'

    if os.path.exists(path_raw):
        print(f'\n[Info] Successfully detect dataset [{dataset}].')
    else:
        raise FileNotFoundError(f'Raw dataset [{dataset}] not found in root path.')

    # read and reindex dataset
    if dataset in ['amazon-book', 'amazon-garden']:
        df_raw = read_amazon(dataset, path_raw)
        df_5_raw = filter_cold_start(df_raw, args.k_cold_start)
    elif dataset == 'gowalla':
        df_raw = read_gowalla(dataset, path_raw)
        df_5_raw = filter_cold_start(df_raw, args.k_cold_start)
    elif dataset == 'yelp2018':
        df_raw = read_yelp2018(dataset, path_raw)
        df_5_raw = filter_cold_start(df_raw, args.k_cold_start)
    else:
        assert dataset == 'lastfm'
        df_5_raw = read_lastfm(dataset, path_raw)

    df_5 = reindex_data(df_5_raw)
    save_data(df_5, path_data)


if __name__ == '__main__':
    main()
