# -*- coding: utf-8 -*-
"""
Practicing basics with a small neural net

@author: gshai
"""

from myneuralnetapi import Layer, Input, Data
from myneuralnetapi import BinaryClassifierNetwork

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt


MODE = {
    'train_network': False,
    'generate_predictions': False,
    'check_bias_variance': False,
    'check_hyperparameters': False,
}

MODE['train_network'] = True

if MODE['generate_predictions'] and not MODE['train_network']:
    MODE['train_network'] = True


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


def _main():
    print('Neural Network App')

    #%% reate/load a dataset

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

    dataset_train = Data(
        'training',
        features=features_train.T,
        labels=labels_train.T,
    )

    dataset_val = Data(
        'validation',
        features=features_val.T,
        labels=labels_val.T,
    )

    dataset_test = Data(
        'testing',
        features=features_test.T,
        no_labels=True,
    )

    assert dataset_train.features.shape[1] \
        + dataset_val.features.shape[1] == m_train + m_val

    # https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib
    plt.figure()
    plt.imshow(features[0, :].reshape((28, 28)), cmap='gray')
    plt.show()

    #%% Build the network

    features_size = dataset_train.n_features
    labels_size = dataset_train.n_labels

    net = BinaryClassifierNetwork()
    net.grad_check()

    net.add_input(Input(features_size))

    net.add_layer(Layer('hidden', 'relu', 200, features_size))
    net.add_layer(Layer('hidden', 'relu', 100, 200))
    net.add_layer(Layer('hidden', 'relu', 50, 100))
    net.add_layer(Layer('hidden', 'relu', 25, 50))
    net.add_layer(Layer('hidden', 'relu', 15, 25))

    net.add_output(Layer('output', 'softmax', labels_size, 15))

    net.show_network()

    #%% Hyperparameters

    epochs = 100000

    learn_rate = 0.01
    learn_decay = 1 / 10000 * 0

    lambd = 1.0 * 0
    threshold = 0.5

    #%% Train the network

    if MODE['train_network']:

        # prediction results will be collected here
        results_table = pd.DataFrame()

        # train routine
        cost_list = net.train(
            input_data=dataset_train,
            epochs=epochs,
            learn_rate=learn_rate,
            learn_decay=learn_decay,
            lambd=lambd,
        )

        results = {'cost_list': cost_list}

        _, prediction_metrics = net.predict_and_compare(
            input_data=dataset_train,
            threshold=threshold,
        )
        results.update(prediction_metrics)

        _, prediction_metrics = net.predict_and_compare(
            input_data=dataset_val,
            threshold=threshold,
        )
        results.update(prediction_metrics)

        results_table = results_table.append(results, ignore_index=True)

        plt.figure()
        plt.plot(cost_list)
        plt.title('Costs (iter / 1000)')
        plt.show()

    #%% Generate predictions

    if MODE['generate_predictions']:
        assert MODE['train_network']

        predictions = net.predict(dataset_test, threshold)
        predictions = net.softmax_to_category(predictions)

        predictions_df = pd.DataFrame(predictions.T).reset_index()
        predictions_df.columns = ['ImageId', 'Label']
        predictions_df['ImageId'] += 1

        predictions_df.to_csv(
            'results/digits_test.csv',
            index=False,
        )

    #%% bias or variance

    if MODE['check_bias_variance']:

        # array with values for various sizes of the training dataset
        size_train_vec = np.linspace(
            start=50,
            stop=dataset_train.n_examples,
            num=10,
            dtype=np.int32,
        )

        # prediction results will be collected here
        results_table = pd.DataFrame()

        for size in size_train_vec:
            # create subset of the dataset
            dataset_train_trimmed = dataset_train.trimmed(size)

            # train routine
            cost_list = net.train(
                input_data=dataset_train_trimmed,
                epochs=epochs,
                learn_rate=learn_rate,
                learn_decay=learn_decay,
                lambd=lambd,
            )

            results = {
                'cost_list': cost_list,
                'size': size,
            }

            _, prediction_metrics = net.predict_and_compare(
                input_data=dataset_train_trimmed,
                threshold=threshold,
            )
            results.update(prediction_metrics)

            _, prediction_metrics = net.predict_and_compare(
                input_data=dataset_val,
                threshold=threshold,
            )
            results.update(prediction_metrics)

            results_table = results_table.append(results, ignore_index=True)

        plt.figure()
        plt.plot(results_table['size'],
                 results_table['training_trimmed_cost'],
                 label='cost_train')
        plt.plot(results_table['size'],
                 results_table['validation_cost'],
                 label='cost_validation')
        plt.legend()
        plt.title('J_train, J_val (m_size)')
        plt.show()

    #%% tune hyperparameters

    if MODE['check_hyperparameters']:

        # array of var learning rates
        learn_rate_vec = np.logspace(
            start=-3,
            stop=0,
            num=12,
            dtype=np.float32,
        )

        # array of var lambdas (regularization)
        lambd_vec = np.logspace(
            start=-3,
            stop=1,
            num=16,
            dtype=np.float32,
        )

        # prediction results will be collected here
        results_table = pd.DataFrame()

        for lambd in lambd_vec:
            for learn_rate in learn_rate_vec:
                # train routine
                cost_list = net.train(
                    input_data=dataset_train,
                    epochs=epochs,
                    learn_rate=learn_rate,
                    learn_decay=learn_decay,
                    lambd=lambd,
                )

                results = {
                    'cost_list': cost_list,
                    'learn_rate': learn_rate,
                    'lambd': lambd,
                }

                _, prediction_metrics = net.predict_and_compare(
                    input_data=dataset_train,
                    threshold=threshold,
                )
                results.update(prediction_metrics)

                _, prediction_metrics = net.predict_and_compare(
                    input_data=dataset_val,
                    threshold=threshold,
                )
                results.update(prediction_metrics)

                results_table = results_table.append(
                    results, ignore_index=True)

        print(results_table)
        results_table.to_csv('result/results_table.csv')


#%%


if __name__ == '__main__':
    _main()