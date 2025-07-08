# coding: utf-8
# @email: enoche.chow@gmail.com
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
import math
import time
from logging import getLogger
import os

import numpy as np
import pandas as pd


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path'] + self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.time_field = self.config['TIME_FIELD'] if self.config["use_time"] else None
        self.splitting_label = self.config['inter_splitting_label']

        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        if self.time_field:
            cols = [self.uid_field, self.iid_field, self.splitting_label, self.time_field]
        else:
            cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        if self.config['cold_start_type'] == 'none':
            return self.warm_split()
        elif self.config['cold_start_type'] in ['user', 'item', 'both']:
            return self.cold_split()
        else:
            raise ValueError('Unknown cold start type: {}'.format(self.config['cold_start_type']))

    def warm_split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)  # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def cold_split(self):
        cold_start_scenario = self.config['cold_start_type']
        ratios = self.config['cold_start_split_ratios']
        rng = np.random.default_rng(self.config['seed'])

        # retrieve unique users, sorted to ensure determinism
        unique_user_indices = sorted(list(self.df[self.uid_field].unique()))
        if cold_start_scenario in ['user', 'both']:
            rng.shuffle(unique_user_indices)
            user_split_indices = self.split_ratio(unique_user_indices, ratios)
        else:
            # use all users in every split
            user_split_indices = (unique_user_indices,) * 3

        # retrieve unique items, sorted to ensure determinism
        unique_item_indices = sorted(list(self.df[self.iid_field].unique()))
        if cold_start_scenario in ['item', 'both']:
            rng.shuffle(unique_item_indices)
            item_split_indices = self.split_ratio(unique_item_indices, ratios)
        else:
            # use all items in every split
            item_split_indices = (unique_item_indices,) * 3

        split_results = []
        for user_indices, item_indices in zip(user_split_indices, item_split_indices):
            temp_df = self.df[self.df[self.uid_field].isin(user_indices) &
                              self.df[self.iid_field].isin(item_indices)].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)  # no use again
            split_results.append(temp_df)

        full_ds = [self.copy(_) for _ in split_results]
        return full_ds

    def split_ratio(self, a, ratios):
        n_samples = len(a)
        n_val = math.ceil(n_samples * ratios[1])
        n_test = math.ceil(n_samples * ratios[2])
        n_train = n_samples - n_val - n_val
        return a[:n_train], a[n_train: n_train + n_val], a[-n_test:]

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num / tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num / tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        if self.time_field:
            uni_time = pd.unique(self.df[self.time_field])
            max_time_span = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max(uni_time)))
            min_time_span = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(min(uni_time)))
            info.extend(['The time span of dataset: {} ~ {}'.format(min_time_span, max_time_span)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
