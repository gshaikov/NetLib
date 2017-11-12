# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:24:08 2017

@author: gshai
"""

from myneuralnetapi import Layer, Input, Data
from myneuralnetapi import BinaryClassifierNetwork

import numpy as np
#import pandas as pd

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
    dataset = Data(
        'boundary',
        features=data.T,
        no_labels=True,
    )

    Z = net.predict(
        input_data=dataset,
        threshold=0.5,
    )

    if net.layers_list[-1].activ_name == 'softmax':
        Z = net.softmax_to_category(Z)

    Z = Z.reshape(xx.shape)

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

    dataset = Data(
        'blobs',
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

    epochs = 30000

    learn_rate = 0.01
    learn_decay = 1 / 5000 * 0

    lambd = 1.0 * 1
    threshold = 0.5

    #%% train the network

    cost_list = net.train(
        input_data=dataset,
        epochs=epochs,
        learn_rate=learn_rate,
        learn_decay=learn_decay,
        lambd=lambd,
    )

    results = {'cost_list': cost_list}

    predictions, prediction_metrics = net.predict_and_compare(
        input_data=dataset,
        threshold=threshold,
    )

    results.update(prediction_metrics)

    plt.figure()
    plt.plot(cost_list)
    plt.title('Costs (iter / 1000)')
    plt.show()

    #%%

    plot_decision_boundary(net, features, labels)


#%%


if __name__ == '__main__':
    _main()
