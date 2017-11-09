# -*- coding: utf-8 -*-
"""
Practicing basics with a small neural net

@author: gshai
"""

from clean_data import create_datasets, normalize_dataset

from myneuralnetapi import Layer, Input, Data
from myneuralnetapi import BinaryClassifierNetwork

#import tensorflow as tf
#from tensorflow.python.framework import ops

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.ERROR)


#####################################################


def main():
    '''main'''
    np.random.seed(1)

    # load engineered dataset
    dataset_main = pd.read_csv('dataset/df.csv')
    dataset_ids = pd.read_csv('dataset/df_id.csv')

    # Split into training, validation, and test datasets
    datasets, labels, sizes = create_datasets(dataset_main, split=0.8)

    # check that dataset was split correctly
    assert sum(list(map((lambda x: x.shape[0]), datasets))) \
        == dataset_main.shape[0]

    # Normalize
    datasets = normalize_dataset(datasets)

    # check that dataset was split correctly
    assert sum(list(map((lambda x: x.shape[0]), datasets))) \
        == dataset_main.shape[0]

    # Create numpy arrays out of DataFrames
    dataset_train = Data(
        dataset_name='training',
        features=datasets[0].drop('PassengerId', axis=1).as_matrix().T,
        labels=labels[0].values.reshape(1, sizes[0])
    )

    dataset_val = Data(
        dataset_name='validation',
        features=datasets[1].drop('PassengerId', axis=1).as_matrix().T,
        labels=labels[1].values.reshape(1, sizes[1])
    )

    dataset_test = Data(
        dataset_name='test',
        features=datasets[2].drop('PassengerId', axis=1).as_matrix().T,
        no_labels=True
    )

    ###############################
    # Build the network

    train_n_features = dataset_train.n_features

    net = BinaryClassifierNetwork()
    net.grad_check()

    net.add_input(Input(train_n_features))

    net.add_layer(Layer('hidden', 'relu', train_n_features, train_n_features))
    net.add_layer(Layer('hidden', 'relu', train_n_features, train_n_features))
    net.add_layer(Layer('hidden', 'relu', train_n_features, train_n_features))
    net.add_layer(Layer('hidden', 'relu', train_n_features, train_n_features))
    net.add_layer(Layer('hidden', 'relu', train_n_features, train_n_features))

    net.add_output(Layer('output', 'sigmoid', 1, train_n_features))

    net.show_network()

    ###############################
    # Hyperparameters

    epochs = 100000

    learn_rate = 0.1
    learn_decay = 1 / 10000 * 0

    lambd = 10.0 * 1
    threshold = 0.5

    ###############################
    # Train the network

    if TRAIN_NETWORK is True:

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

    ###############################
    # Generate predictions

    if GENERATE_PREDICTIONS is True:
        assert TRAIN_NETWORK

        predictions = net.predict(dataset_test, threshold)

        predicted_ids = pd.concat([
            dataset_ids[
                dataset_ids['Train'] == 0][
                    'PassengerId'].reset_index(drop=True),
            pd.Series(np.squeeze(predictions.T)),
        ], axis=1)

        predicted_ids.to_csv(
            'result/output.csv',
            header=['PassengerId', 'Survived'],
            index=False,
        )

    ###############################
    # Diagnose network - bias or variance
    # Train network with different sizes of the training dataset

    if CHECK_BIAS_VARIANCE is True:

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

    ###############################
    # Diagnose network - tune hyperparameters

    if CHECK_HYPERPARAMETERS is True:

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

    ###############################
    # Use tensorflow

    if USE_TENSORFLOW_INSTEAD is True:
        pass

    return results_table


#####################################################


if __name__ == '__main__':
    TRAIN_NETWORK = True
    GENERATE_PREDICTIONS = False

    CHECK_BIAS_VARIANCE = False
    CHECK_HYPERPARAMETERS = False

    USE_TENSORFLOW_INSTEAD = False

    if GENERATE_PREDICTIONS and not TRAIN_NETWORK:
        TRAIN_NETWORK = True

    RESULTS_DF = main()
