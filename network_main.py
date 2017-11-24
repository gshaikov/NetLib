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

import IPython.display

import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
def save_parameters(model, filename):
    '''save_parameters'''
    layers = model.layers_list
    parameters = dict()
    for idx, lay in enumerate(layers):
        if isinstance(lay, Layer):  # skip input layer
            parameters['layer_' + str(idx)] = {
                'layer_name': lay.layer_name,
                'activ_name': lay.activ_name,
                'n_units': lay.n_units,
                'n_connections': lay.n_connections,
                'weight': lay.weight,
                'bias': lay.bias,
            }
    with open(filename, 'wb') as outputfile:
        pickle.dump(parameters, outputfile, pickle.HIGHEST_PROTOCOL)


def load_parameters(filename):
    '''load_parameters'''
    with open(filename, 'rb') as inputfile:
        parameters = pickle.load(inputfile)
    return parameters


def _main():
    print('Neural Network App')

    #%% Load run mode parameters

    # https://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file
    with open('params_mode.json', 'rb') as jsonfile:
        params_mode = json.load(jsonfile)

    #%% Build the network

    features_size = 784
    labels_size = 10

    BinaryClassifierNetwork.grad_check()

    net = BinaryClassifierNetwork()
    net.add_input(Input(features_size))

    if not params_mode['load_parameters']:
        print("\nCreating the model...")
        net.add_layer(Layer('hidden', 'relu', 400, features_size))
        net.add_layer(Layer('hidden', 'relu', 200, 400))
        net.add_layer(Layer('hidden', 'relu', 100, 200))
        net.add_layer(Layer('hidden', 'relu', 50, 100))
        net.add_layer(Layer('hidden', 'relu', 25, 50))
        net.add_output(Layer('output', 'softmax', labels_size, 25))

    else:
        print("\nLoading the model...")
        parameters = load_parameters('tmp/network_parameters_00.pkl')
        for layer_no, params in parameters.items():
            print("Loading {}, with {}".format(
                layer_no, params['activ_name']))
            net.add_layer(Layer(
                params['layer_name'],
                params['activ_name'],
                params['n_units'],
                params['n_connections'],
            ))
            net.layers_list[-1].weight = params['weight']
            net.layers_list[-1].bias = params['bias']

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

    epochs = params_mode['epochs']
    minibatches_size = params_mode['minibatches_size']
    learn_rate = params_mode['learning_rate']
    lambd_val = params_mode['lambda_reg']

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

    if not params_mode['load_parameters']:

        costs_run_table = pd.DataFrame(columns=[
            'cost_train',
            'accuracy_train',
            'cost_dev',
            'accuracy_dev',
        ])

        net.reset_network()

        try:
            for epoch in range(1, epochs + 1):

                dataset.shuffle_train()
                train_batches = dataset.get_train_batches(minibatches_size)

                if epoch == 0:
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
                    plt.close(fig)

                number_of_iterations = len(train_batches)

                for idx, (examples_features, examples_labels) in enumerate(train_batches):

                    dataset_train_batch = DataInNetwork(
                        features=examples_features,
                        labels=examples_labels,
                    )
                    net.load_data(dataset_train_batch)

                    # train routine
                    net.train(learn_rate=learn_rate, lambd=lambd_val)

                    _, prediction_metrics_train_batch \
                        = net.predict_and_compare(dataset_train_batch)

                    if idx % 10 == 0:
                        IPython.display.clear_output(wait=True)
                        print("epoch {:3d}, minibatch {:4d}/{:4d}, cost {:5.4f}, acc {:6.4f}%".format(
                            epoch, idx + 1, number_of_iterations,
                            prediction_metrics_train_batch['cost'],
                            prediction_metrics_train_batch['accuracy'] * 100,
                        ), end="\r")

                _, prediction_metrics_train \
                    = net.predict_and_compare(dataset_train)

                _, prediction_metrics_dev \
                    = net.predict_and_compare(dataset_dev)

                costs_run_table = costs_run_table.append({
                    'epoch': epoch,
                    'cost_train': prediction_metrics_train['cost'],
                    'accuracy_train': prediction_metrics_train['accuracy'] * 100,
                    'cost_dev': prediction_metrics_dev['cost'],
                    'accuracy_dev': prediction_metrics_dev['accuracy'] * 100,
                }, ignore_index=True)

                print("\nepoch {:3d}; cost train {:5.4f}, dev {:5.4f}; acc train {:6.4f}%, dev {:6.4f}%".format(
                    epoch,
                    prediction_metrics_train['cost'],
                    prediction_metrics_dev['cost'],
                    prediction_metrics_train['accuracy'] * 100,
                    prediction_metrics_dev['accuracy'] * 100,
                ))

        except KeyboardInterrupt:
            pass

        fig = plt.figure()
        plt.plot(costs_run_table['cost_train'])
        plt.plot(costs_run_table['cost_dev'])
        plt.legend(['cost_train', 'cost_dev'])
        plt.title('Costs (epoch)')
        # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib
        fig.savefig('results/costs_epoch.png')
        plt.show()
        plt.close(fig)

        fig = plt.figure()
        plt.plot(costs_run_table['accuracy_train'])
        plt.plot(costs_run_table['accuracy_dev'])
        plt.legend(['accuracy_train', 'accuracy_dev'])
        plt.title('Accuracy (epoch)')
        fig.savefig('results/accuracy_epoch.png')
        plt.show()
        plt.close(fig)

    #%% Final model evaluation

    _, prediction_metrics_train \
        = net.predict_and_compare(dataset_train)
    _, prediction_metrics_dev \
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

    if params_mode['generate_predictions']:

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
            fig = plt.figure()
            plt.imshow(
                test_features_array[:, idx].reshape((28, 28)), cmap='gray')
            plt.show()
            plt.close(fig)
            print("predicted: {}\n".format(row['Label']))

        predictions_test_df.to_csv(
            'results/digits_test.csv',
            index=False,
        )

    #%% Save the model

    if not params_mode['load_parameters']:
        print("Saving the model...")
        save_parameters(net, 'tmp/network_parameters_00.pkl')


#%%


if __name__ == '__main__':
    _main()
