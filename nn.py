# -*- coding: utf-8 -*-
"""
Practicing basics with a small neural net

@author: gshai
"""

from clean_data import create_datasets, normalize_dataset

from myneuralnetapi import Layer, Input, Data
from myneuralnetapi import BinaryClassifierNetwork

import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

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
    datasets = normalize_dataset(datasets[0], datasets)

    # check that dataset was split correctly
    assert sum(list(map((lambda x: x.shape[0]), datasets))) \
        == dataset_main.shape[0]

    # Create numpy arrays out of DataFrames
    dataset_train = Data(
        dataset_name='training',
        features=datasets[0].as_matrix().T,
        labels=labels[0].values.reshape(1, sizes[0])
    )

    dataset_val = Data(
        dataset_name='validation',
        features=datasets[1].as_matrix().T,
        labels=labels[1].values.reshape(1, sizes[1])
    )

    dataset_test = Data(
        dataset_name='test',
        features=datasets[2].as_matrix().T,
        no_labels=True
    )

    ###############################
    # Build the network

    train_n_features = dataset_train.n_features

    net = BinaryClassifierNetwork()
    net.grad_check()

    net.add_input(Input(train_n_features))
    net.add_layer(Layer('hidden', 'relu', 20, train_n_features))
    net.add_layer(Layer('hidden', 'relu', 10, 20))
#    net.add_layer(Layer('hidden', 'relu', train_n_features, train_n_features))
    net.add_output(Layer('output', 'sigmoid', 1, 10))

    net.show_network()

    ###############################
    # Hyperparameters

    epochs = 100000
    learn_rate = 0.1
    learn_decay = 1 / 10000

    lambd = 1.0
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
            num=4,
            dtype=np.float32,
        )

        # array of var lambdas (regularization)
        lambd_vec = np.logspace(
            start=-2,
            stop=0,
            num=5,
            dtype=np.float32,
        )

        # prediction results will be collected here
        results_table = pd.DataFrame()

        for learn_rate in learn_rate_vec:
            for lambd in lambd_vec:

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

    if USE_TENSORFLOW_INSTEAD == True:
        ops.reset_default_graph()

        X = tf.placeholder(
            dtype=tf.float32, shape=(train_n_features, None), name='X')
        Y = tf.placeholder(
            dtype=tf.float32, shape=(1, None), name='Y')

        tf.set_random_seed(1)

        seed = 1

        W1 = tf.get_variable(
            "W1", [20, train_n_features], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b1 = tf.get_variable(
            "b1", [20, 1], initializer=tf.zeros_initializer())
        W2 = tf.get_variable(
            "W2", [10, 20], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b2 = tf.get_variable(
            "b2", [10, 1], initializer=tf.zeros_initializer())
        W3 = tf.get_variable(
            "W3", [1, 10], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b3 = tf.get_variable(
            "b3", [1, 1], initializer=tf.zeros_initializer())

        parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3,
        }

        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)

        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learn_rate).minimize(cost)

        init = tf.global_variables_initializer()

        seed = 3

        costs = []

        with tf.Session() as sess:
            sess.run(init)

            total_cost = 0.

            for epoch in range(epochs):
                _, curr_cost = sess.run(
                    [optimizer, cost],
                    feed_dict={X: datasets[0], Y: labels[0]})

                total_cost += curr_cost

                if epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, curr_cost))
                if epoch % 100 == 0:
                    costs.append(curr_cost)

            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learn_rate))
            plt.show()

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval(
                {X: datasets[0], Y: labels[0]}))
            print("Test Accuracy:", accuracy.eval(
                {X: datasets[2], Y: labels[2]}))

    return results_table


#####################################################


if __name__ == '__main__':
    TRAIN_NETWORK = False
    CHECK_BIAS_VARIANCE = False
    CHECK_HYPERPARAMETERS = False
    GENERATE_PREDICTIONS = False

    USE_TENSORFLOW_INSTEAD = True

    if GENERATE_PREDICTIONS and not TRAIN_NETWORK:
        TRAIN_NETWORK = True

    RESULTS_DF = main()
