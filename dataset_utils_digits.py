# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:25:57 2017

@author: gshai
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def take_features(dataset):
    '''take_features'''
    # dataset to array, normalize features
    features = dataset[['pixel' + str(idx) for idx in range(784)]].as_matrix()
    features = features / (np.max(features) - np.min(features))
    return features


def split_array(data_array, *args):
    '''dataset of variable size data arrays'''
    if args:
        sets = list()
        start = 0
        for size in args:
            sets.append(data_array[start:start + size, :])
            start = size
    else:
        sets = [data_array]
    assert isinstance(sets, list)
    return sets


def get_digit_dataset(m_train, m_val):
    '''get_digit_dataset'''
    csv_table = pd.read_csv('dataset/train.csv')
    csv_table_test = pd.read_csv('dataset/test.csv')

    # shuffle training dataframe
    csv_table = csv_table.sample(frac=1).reset_index(drop=True)

    # create features arrays
    features = take_features(csv_table)
    features_test = take_features(csv_table_test)

    # create labels array, convert 0...9 labels to [0,1,0,...,0] vectors
    labels = csv_table['label'].as_matrix()[:, np.newaxis]
    onehot_encoder = OneHotEncoder(sparse=False)
    labels_onehot = onehot_encoder.fit_transform(labels)

    # sizes of datasets
    m_train = 1000
    m_val = 1000

    # split arrays into train and val
    features_train, features_val = split_array(features, m_train, m_val)
    labels_train, labels_val = split_array(labels_onehot, m_train, m_val)

    result = {
        'features_train': features_train.T,
        'features_val': features_val.T,
        'features_test': features_test.T,
        'labels_train': labels_train.T,
        'labels_val': labels_val.T,
    }

    return result


if __name__ == '__main__':
    get_digit_dataset(1, 1)
