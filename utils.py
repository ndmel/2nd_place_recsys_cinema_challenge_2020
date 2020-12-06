import pandas as pd
import numpy as np
import scipy
import datetime
import tqdm
from utils import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

user_col = 'cont_user_id'
movie_col = 'cont_tv_show_id'

def apk(actual, predicted, k=5):
    
    '''
    Function to get Average Precision at K
    '''
    
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=5):
    
    '''
    Function to get Mean Average Precision at K
    '''
    
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def get_features(df, get_n = 5, threshold = 0.5):
    
    '''
    Function to get target-like features by threshold
    '''
    
    # filter by 80% watch time
    df = df[df['user_watch_perc'] > threshold].copy()
    
    # calc how many times user viewed the show
    df_views = df[['user_id', 'tv_show_id']].groupby(['user_id', 'tv_show_id']).size().reset_index()\
        .rename(columns = {0:'user_show_freq'})
    
    # get only top-5 for each user
    df_top_views = df_views.sort_values(['user_id', 'user_show_freq'], ascending=False).groupby(['user_id']).head(get_n)
    
    return df_top_views


def get_target(df, get_n = 5):
    
    '''
    Function to get target
    '''
    
    # filter by 80% watch time
    df = df[df['user_watch_perc'] >= 0.8].copy()
    
    # calc how many times user viewed the show
    df_views = df[['user_id', 'tv_show_id']].groupby(['user_id', 'tv_show_id']).size().reset_index()\
        .rename(columns = {0:'user_show_freq'})
    
    # get only top-5 for each user
    df_top_views = df_views.sort_values(['user_id', 'user_show_freq'], ascending=False).groupby(['user_id']).head(get_n)
    
    return df_top_views

def prepare_features(train, train_info, train_info_future, train_base):
    
    '''
    Function to enrich data by adding features
    '''

    final_shape = train.shape[0]
    
    # collect prediction rank
    for user, group in train.sort_values(['user_id', 'user_show_freq'], ascending = False).groupby(['user_id']):
        train.loc[group.index, 'user_show_rank'] = np.arange(0, group.shape[0])
    train.loc[train[train.user_show_freq == 0].index, 'user_show_rank'] = 50

    # collect prediction rank
    for user, group in train.sort_values(['user_id', 'lightfm_score'], ascending = False).groupby(['user_id']):
        train.loc[group.index, 'user_lfm_rank'] = np.arange(0, group.shape[0])
    train.loc[train[pd.isnull(train.lightfm_score)].index, 'user_lfm_rank'] = 50

    # mean rank
    train['combined_rank'] = (train['user_show_rank'] + train['user_lfm_rank']) / 2

    # tv show channel
    _temp = train_info[['tv_show_id', 'channel_id']].drop_duplicates()
    _temp = _temp[~_temp.duplicated(['tv_show_id'], keep = 'first')].copy()
    train = train.merge(_temp, on = ['tv_show_id'], how = 'left')
    assert final_shape == train.shape[0]

    # add cat ids
    _temp = train_info[['tv_show_id', 'tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3']].drop_duplicates()
    _temp = _temp[~_temp.duplicated(['tv_show_id'], keep = 'first')].copy()
    train = train.merge(_temp, on = ['tv_show_id'], how = 'left')
    assert final_shape == train.shape[0]
    
    # number of watches with diff thresholds
    for threshold in [0.3, 0.8]:
        c_new = f'user_show_freq_{threshold}'
        features_threshold = get_features(train_base, threshold=threshold)
        features_threshold.rename(columns = {'user_show_freq':c_new}, inplace = True)
        train = train.merge(features_threshold, on = ['user_id', 'tv_show_id'], how = 'left')
        train[c_new].fillna(0, inplace = True)
        assert final_shape == train.shape[0]
        
    # alternative base
    total_user_show_watch_df = train_base.groupby(['tv_show_id', 'show_start_time', 'show_stop_time', 'user_id', 'show_duration'], as_index = False).user_watch_time.sum()
    total_user_show_watch_df['user_watch_perc'] = total_user_show_watch_df['user_watch_time'] / total_user_show_watch_df['show_duration']
    total_user_show_watch_df = total_user_show_watch_df[total_user_show_watch_df.user_watch_perc <= 1].copy()
    
    # number of watches with diff thresholds
    for threshold in [0.3, 0.5, 0.8]:
        c_new = f'alt_user_show_freq_{threshold}'
        features_threshold = get_features(total_user_show_watch_df, threshold=threshold)
        features_threshold.rename(columns = {'user_show_freq':c_new}, inplace = True)
        train = train.merge(features_threshold, on = ['user_id', 'tv_show_id'], how = 'left')
        train[c_new].fillna(0, inplace = True)
        assert final_shape == train.shape[0]
    

    # number of watches with more recent time splits
    for weeks_prior in [1]:
        c_new = f'user_show_freq_week_{weeks_prior}'
        split_date = train_base.start_time.max() - datetime.timedelta(days = weeks_prior * 7)
        features_split = get_features(train_base[train_base.start_time >= split_date], 200)
        features_split.rename(columns = {'user_show_freq':c_new}, inplace = True)
        train = train.merge(features_split, on = ['user_id', 'tv_show_id'], how = 'left')
        train[c_new].fillna(0, inplace = True)
        assert final_shape == train.shape[0]
        
        train[f'user_show_freq_dif_week_{weeks_prior}'] = (train['user_show_freq'] - train[c_new]) / train['user_show_freq']
        assert final_shape == train.shape[0]
        
        # new ranks
        for user, group in train.sort_values(['user_id', f'user_show_freq_week_{weeks_prior}'], ascending = False).groupby(['user_id']):
            train.loc[group.index, f'user_show_rank_week_{weeks_prior}'] = np.arange(0, group.shape[0])
        train[f'user_show_rank_mean_week_{weeks_prior}'] = train[['user_show_rank', f'user_show_rank_week_{weeks_prior}']].mean(axis = 1)
        assert final_shape == train.shape[0]
        
        train.drop(c_new, 1, inplace = True)
    
    # number of watches with more recent time splits
    for weeks_prior in [1]:
        c_new = f'alt_user_show_freq_week_{weeks_prior}'
        split_date = total_user_show_watch_df.show_start_time.max() - datetime.timedelta(days = weeks_prior * 7)
        features_split = get_features(total_user_show_watch_df[total_user_show_watch_df.show_start_time >= split_date], 200)
        features_split.rename(columns = {'user_show_freq':c_new}, inplace = True)
        train = train.merge(features_split, on = ['user_id', 'tv_show_id'], how = 'left')
        train[c_new].fillna(0, inplace = True)

        train[f'alt_user_show_freq_dif_week_{weeks_prior}'] = (train['user_show_freq'] - train[c_new]) / train['user_show_freq']

        # new ranks
        for user, group in train.sort_values(['user_id', c_new], ascending = False).groupby(['user_id']):
            train.loc[group.index, f'user_show_rank_week_{weeks_prior}'] = np.arange(0, group.shape[0])
        train[f'alt_user_show_rank_mean_week_{weeks_prior}'] = train[['user_show_rank', f'user_show_rank_week_{weeks_prior}']].mean(axis = 1)

        train.drop(c_new, 1, inplace = True)

    # Насколько часто пользователь смотрит канал
    _temp = train.groupby(['user_id', 'channel_id']).size().reset_index().rename(columns = {0:'user_channel_count'})
    _temp = _temp.merge(_temp.groupby(['user_id'], as_index = False)['user_channel_count'].sum().rename(columns = {'user_channel_count':'user_count'}),
                on = ['user_id'], how = 'left')
    train = train.merge(_temp[['user_id', 'channel_id', 'user_channel_count']],
                on = ['user_id', 'channel_id'], how = 'left')
    
    # user gruop watch mean time+ rel to every watch
    train = train.merge(train.groupby(['user_id'], as_index = False)['user_show_freq'].mean().rename(columns = {'user_show_freq':'group_user_show_freq'}),
                on = ['user_id'], how = 'left')
    train['user_show_freq_rel_group'] = train['user_show_freq'] / train['group_user_show_freq']
    assert final_shape == train.shape[0]

    # show total duration in the future and relative to previous
    _temp_1 = train_info.groupby(['tv_show_id'], as_index = False)['duration'].sum().rename(columns = {'duration':'tot_show_duration'})
    num_days = (train_info.start_time.max() - train_info.start_time.min()).days
    _temp_1['tot_show_duration'] /= num_days
    _temp_2 = train_info_future.groupby(['tv_show_id'], as_index = False)['duration'].sum().rename(columns = {'duration':'tot_show_duration_future'})
    num_days = (train_info_future.start_time.max() - train_info_future.start_time.min()).days
    _temp_2['tot_show_duration_future'] /= num_days
    train = train.merge(_temp_1, on = ['tv_show_id'], how = 'left')
    train = train.merge(_temp_2, on = ['tv_show_id'], how = 'left')
    train['tot_show_duration'].fillna(0, inplace = True)
    train['tot_show_duration_future'].fillna(0, inplace = True)
    train['popularity_drop'] = train['tot_show_duration_future'] / train['tot_show_duration']
    assert final_shape == train.shape[0]
    
    return train


def df_to_sparse_interaction_matrix(x, has_seen = True):
    '''
    Pandas dataframe to LightFM format
    '''

    interaction_x = x[[user_col, movie_col]].drop_duplicates().assign(seen=1).\
                    pivot_table(index = user_col, columns = movie_col).fillna(0)
        
    return scipy.sparse.csr_matrix(interaction_x)