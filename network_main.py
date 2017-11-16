# -*- coding: utf-8 -*-
"""
Practicing basics with a small neural net

@author: gshai
"""

import pickle
import json

from myneuralnetapi import Layer, Input, DataInNetwork
from myneuralnetapi import BinaryClassifierNetwork
from datahandler import DataContainer

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
def save_model(model, filename):
    '''save_model'''
    model.load_data(DataInNetwork(
        features=np.array([[1, 2]]),
        labels=np.array([[1, 2]])
    ))
    with open(filename, 'wb') as outputfile:
        pickle.dump(model, outputfile, pickle.HIGHEST_PROTOCOL)


def load_model(filename):
    '''load_model'''
    with open(filename, 'rb') as inputfile:
        model = pickle.load(inputfile)
    return model


def _main():
    print('Neural Network App')

    #%% Load run mode parameters

    # https://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file
    with open('run_params.json', 'rb') as jsonfile:
        run_params = json.load(jsonfile)

    #%% Build the network

    features_size = 784
    labels_size = 10

    BinaryClassifierNetwork.grad_check()

    if not run_params['load_model']:
        print("\nCreating the model...")
        net = BinaryClassifierNetwork()

        net.add_input(Input(features_size))

        net.add_layer(Layer('hidden', 'relu', 400, features_size))
        net.add_layer(Layer('hidden', 'relu', 200, 400))
        net.add_layer(Layer('hidden', 'relu', 100, 200))
        net.add_layer(Layer('hidden', 'relu', 50, 100))
        net.add_layer(Layer('hidden', 'relu', 25, 50))

        net.add_output(Layer('output', 'softmax', labels_size, 25))

    else:
        print("\nLoading the model...")
        net = load_model('tmp/network_model_00.pkl')

    net.show_network()

    #%% reate/load a dataset

    # sizes of datasets
    m_train = 40000
    m_dev = 2000

    dataset = DataContainer.get_dataset('dataset/train.csv', shuffle=True)
    dataset.split_train_dev(m_train, m_dev)

    dataset_train = DataInNetwork(
        features=dataset.train_features,
        labels=dataset.train_labels,
    )

    dataset_dev = DataInNetwork(
        features=dataset.dev_features,
        labels=dataset.dev_labels,
    )

    #%% Hyperparameters

    epochs = 200
    minibatches_size = 2048

    learn_rate = 0.0001

    lambd_val = 3000.0

    #%% Train the network

    # prediction results will be collected here
    results_final_model = pd.DataFrame(columns=[
        'model',
        'lambda',
        'keep_prob',
        'cost_train',
        'accuracy_train',
        'cost_dev',
        'accuracy_dev',
    ])

    net.reset_network()

    costs_run_table = pd.DataFrame(columns=[
        'cost_train',
        'accuracy_train',
        'cost_dev',
        'accuracy_dev',
    ])

    try:
        for epoch in range(1, epochs + 1):

            dataset.shuffle_train()
            train_batches = dataset.get_train_batches(minibatches_size)

            if run_params['enable_plots']:
                # https://stackoverflow.com/questions/25239933/how-to-add-title-to-subplots-in-matplotlib
                # https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib
                # https://stackoverflow.com/questions/39659998/using-pyplot-to-create-grids-of-plots
                fig = plt.figure()

                ax1 = fig.add_subplot(1, 2, 1)
                digits_train_example = train_batches[0][0][:, 0]
                ax1.imshow(digits_train_example.reshape(
                    (28, 28)), cmap='gray')
                ax1.set_title("train [0]")

                ax2 = fig.add_subplot(1, 2, 2)
                digits_train_example = dataset.dev_features[:, 0]
                ax2.imshow(digits_train_example.reshape(
                    (28, 28)), cmap='gray')
                ax2.set_title("dev [0]")

                plt.show()

            cost_train_epoch = 0
            number_of_iterations = len(train_batches)

            for examples_features, examples_labels in train_batches:

                dataset_train_batch = DataInNetwork(
                    features=examples_features,
                    labels=examples_labels,
                )
                net.load_data(dataset_train_batch)

                # train routine
                cost_train_batch = net.train(
                    learn_rate=learn_rate,
                    lambd=lambd_val,
                )

                cost_train_epoch += cost_train_batch / number_of_iterations

            predictions, prediction_metrics_train \
                = net.predict_and_compare(dataset_train)

            net.load_data(dataset_dev)
            net.forward_prop()
            cost_dev_epoch = net.calc_cost()

            predictions, prediction_metrics_dev \
                = net.predict_and_compare(dataset_dev)

            costs_run_table = costs_run_table.append({
                'epoch': epoch,
                'cost_train': cost_train_epoch,
                'accuracy_train': prediction_metrics_train['accuracy'] * 100,
                'cost_dev': cost_dev_epoch,
                'accuracy_dev': prediction_metrics_dev['accuracy'] * 100,
            }, ignore_index=True)

            print(
                "epoch {}; cost train {:5.4f}, dev {:5.4f}; acc train {:6.4f}%, dev {:6.4f}%".format(
                    epoch,
                    cost_train_epoch,
                    cost_dev_epoch,
                    prediction_metrics_train['accuracy'] * 100,
                    prediction_metrics_dev['accuracy'] * 100))

    except KeyboardInterrupt:
        pass

    if run_params['enable_plots']:
        plt.figure()
        plt.plot(costs_run_table['cost_train'])
        plt.plot(costs_run_table['cost_dev'])
        plt.legend(['cost_train', 'cost_dev'])
        plt.title('Costs (epoch)')
        plt.show()

        plt.figure()
        plt.plot(costs_run_table['accuracy_train'])
        plt.plot(costs_run_table['accuracy_dev'])
        plt.legend(['accuracy_train', 'accuracy_dev'])
        plt.title('Accuracy (epoch)')
        plt.show()

    predictions, prediction_metrics_train \
        = net.predict_and_compare(dataset_train)
    predictions, prediction_metrics_dev \
        = net.predict_and_compare(dataset_dev)

    print("Final accuracy\ntrain: {:6.4f}%\ndev:   {:6.4f}%".format(
        prediction_metrics_train['accuracy'] * 100,
        prediction_metrics_dev['accuracy'] * 100))

    results_final_model = results_final_model.append({
        'model': 'model_00',
        'lambda': lambd_val,
        'keep_prob': 1.0,
        'cost_train': prediction_metrics_train['cost'],
        'accuracy_train': prediction_metrics_train['accuracy'],
        'cost_dev': prediction_metrics_dev['cost'],
        'accuracy_dev': prediction_metrics_dev['accuracy'],
    }, ignore_index=True)

    results_final_model.to_csv(
        'tmp/hyperparameter_search.csv',
        index=True,
    )

    #%% Generate predictions

    if run_params['generate_predictions']:

        input("Press Enter to Predict...")

        dataset_test = DataContainer.get_dataset('dataset/test.csv')
        test_features_array = dataset_test.data.T

        dataset_test = DataInNetwork(
            features=test_features_array,
            no_labels=True,
        )

        predictions_test = net.predict(dataset_test).T

        predictions_test_df = pd.DataFrame(predictions_test).reset_index()
        predictions_test_df.columns = ['ImageId', 'Label']
        predictions_test_df['ImageId'] += 1

        for idx, row in predictions_test_df.head(10).iterrows():
            plt.figure()
            plt.imshow(
                test_features_array[:, idx].reshape((28, 28)), cmap='gray')
            plt.show()
            print("predicted: {}\n".format(row['Label']))

        predictions_test_df.to_csv(
            'results/digits_test.csv',
            index=False,
        )

    #%% Save the model

    print("Saving the model...")
    save_model(net, 'tmp/network_model_00.pkl')


#%%


if __name__ == '__main__':
    _main()
