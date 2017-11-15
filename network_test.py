# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:24:08 2017

@author: gshai
"""

from myneuralnetapi import Layer, Input, DataInNetwork
from myneuralnetapi import BinaryClassifierNetwork

import numpy as np
import pandas as pd

from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt


# From: https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(net, X, y):
    '''plot_decision_boundary'''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict the function value for the whole gid
    data = np.c_[xx.ravel(), yy.ravel()]
    dataset = DataInNetwork(
        features=data.T,
        no_labels=True,
    )

    Z = net.predict(dataset).reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(y), cmap=plt.cm.Spectral)
    plt.show()


def _main():
    print("Neural Network Test")

    #%% generate dataset

    # http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
    data = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=4,
        cluster_std=1.0,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=None
    )

    features = data[0]
    labels = data[1][:, np.newaxis]

    onehot_encoder = OneHotEncoder(sparse=False)
    labels_onehot = onehot_encoder.fit_transform(labels)

    dataset = DataInNetwork(
        features=features.T,
        labels=labels_onehot.T,
    )

    #%% plot data

    plt.figure()
    plt.scatter(features[:, 0], features[:, 1], c=np.squeeze(labels))
    plt.show()

    #%% build the network

    features_size = dataset.n_features
    labels_size = dataset.n_labels

    BinaryClassifierNetwork.grad_check()

    net = BinaryClassifierNetwork()

    net.add_input(Input(features_size))

    net.add_layer(Layer('hidden', 'relu', 10, features_size))
    net.add_layer(Layer('hidden', 'relu', 10, 10))
    net.add_layer(Layer('hidden', 'relu', 10, 10))
    net.add_layer(Layer('hidden', 'relu', 10, 10))

    net.add_output(Layer('output', 'softmax', labels_size, 10))

    net.show_network()

    #%% Hyperparameters

    epochs = 10000

    learn_rate = 0.001

    lambd_val = 300.0

    #%% train the network

    net.reset_network()

    costs_run_table = pd.DataFrame(columns=['cost_train'])

    try:
        net.load_data(dataset)

        for epoch in range(1, epochs + 1):
            # train routine
            cost_train_epoch = net.train(
                learn_rate=learn_rate,
                lambd=lambd_val,
            )

            costs_run_table = costs_run_table.append({
                'cost_train': cost_train_epoch,
            }, ignore_index=True)

            if epoch % 100 == 0:
                print("epoch {} train cost {:.4f}".format(
                    epoch, cost_train_epoch))

    except KeyboardInterrupt:
        pass

    predictions, prediction_metrics_train \
        = net.predict_and_compare(dataset)

    print("Accuracy train: {:6.4f}%".format(
        prediction_metrics_train['accuracy'] * 100))

    plt.figure()
    plt.plot(costs_run_table['cost_train'])
    plt.legend(['cost_train'])
    plt.title('Costs (epoch)')
    plt.show()

    #%%

    plot_decision_boundary(net, features, labels)


#%%


if __name__ == '__main__':
    _main()
