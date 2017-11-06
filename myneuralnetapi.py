# -*- coding: utf-8 -*-
"""
My custom neural net API

@author: gshai
"""

import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.ERROR)


#####################################################


class Layer(object):
    '''
    Hidden and output layer
    Activations:
    * relu
    * sigmoid
    '''

    def __init__(self, layer_name, activ_name, n_units, n_connections):
        self.weight = None
        self.bias = None

        self.d_weight = None
        self.d_bias = None

        self.linear = None
        self.activation = None

        self.d_linear = None
        self.d_activation = None

        if not isinstance(layer_name, str):
            raise ValueError
        self.layer_name = layer_name

        if not isinstance(activ_name, str):
            raise ValueError
        self.activ_name = activ_name

        if not (isinstance(n_units, int) or n_units > 0):
            raise ValueError
        self.n_units = n_units

        if not (isinstance(n_connections, int) or n_connections > 0):
            raise ValueError
        self.n_connections = n_connections

        self.reset_weights()

    def reset_weights(self):
        '''reset_weights'''
        self.weight = np.random.randn(self.n_units, self.n_connections)
        self.bias = np.zeros((self.n_units, 1))

        if self.activ_name == 'relu':
            self.weight *= np.sqrt(2 / self.n_connections)
        elif self.activ_name == 'sigmoid':
            self.weight *= np.sqrt(1 / self.n_connections)
        else:
            raise Exception("Activation function not specified")

    def activate(self):
        '''activate'''
        if self.activ_name == 'relu':
            self.activation = self.relu(self.linear)
        elif self.activ_name == 'sigmoid':
            self.activation = self.sigm(self.linear)
        else:
            raise Exception("Can't activate this layer")

    def derivate(self):
        '''activate'''
        if self.activ_name == 'relu':
            return self.drelu(self.linear)
        elif self.activ_name == 'sigmoid':
            return self.dsigm(self.linear)
        else:
            raise Exception("Can't derivate this layer")

    @staticmethod
    def relu(lin_z):
        '''Relu'''
        return lin_z * (lin_z > 0)

    @staticmethod
    def drelu(lin_z):
        '''dRelu'''
        return np.array(lin_z > 0, dtype=np.float32)

    @staticmethod
    def sigm(lin_z):
        '''Sigm'''
        # apply function to elements in the array
        # https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
        v_upper = np.vectorize(lambda x: 20 if x > 20 else x)
        v_lower = np.vectorize(lambda x: -20 if x < -20 else x)
        lin_z = v_upper(lin_z)
        lin_z = v_lower(lin_z)
        val = np.exp(lin_z) / (1 + np.exp(lin_z))
        return val

    def dsigm(self, lin_z):
        '''dSigm'''
        sigmoid_activation = self.sigm(lin_z)
        return sigmoid_activation * (1 - sigmoid_activation)


class Input(object):
    '''Input layer'''

    def __init__(self, n_units):
        self.activation = None

        if not (isinstance(n_units, int) or n_units > 0):
            raise ValueError
        self.n_units = n_units

        self.layer_name = 'input'


class Data(object):
    '''
    Data
    features and labels are numpy.ndarray type
    * features have dimentions n by m, where
    * * n - number of features
    * * m - number of data examples
    * labels have dimentions 1 by m
    '''

    def __init__(self,
                 dataset_name,
                 features=np.array([[]]),
                 labels=np.array([[]]),
                 no_labels=False):
        assert isinstance(features, np.ndarray)
        assert features.shape[0] <= features.shape[1]

        self.dataset_name = dataset_name
        self.features = features
        self.labels = None

        self.n_features = features.shape[0]
        self.n_examples = features.shape[1]

        if no_labels is False:
            assert isinstance(labels, np.ndarray)
            assert labels.shape[0] <= labels.shape[1]

            # ensure one output label per example
            assert labels.shape[0] == 1

            # number of data examples is the same
            assert features.shape[1] == labels.shape[1]

            self.labels = labels

    def trimmed(self, size):
        '''trim the dataset so it becomes shorter'''
        new_features = self.features[:, :size]
        if not self.labels is None:
            new_labels = self.labels[:, :size]
            return Data(
                dataset_name=self.dataset_name + '_trimmed',
                features=new_features,
                labels=new_labels
            )
        else:
            return Data(
                dataset_name=self.dataset_name + '_trimmed',
                features=new_features,
                no_labels=True
            )


#####################################################


class BinaryClassifierNetwork(object):
    '''Network'''

    def __init__(self):
        self.layers_list = list()
        self.data = None

    def load_data(self, input_data):
        '''
        load_data
        input_data : Data
        '''
        if not isinstance(input_data, Data):
            raise ValueError('Pass Data object')

        if not isinstance(self.layers_list[0], Input):
            raise ValueError('Create input layer first')

        self.data = input_data
        self.layers_list[0].activation = input_data.features

    def add_input(self, layer):
        '''
        add_input
        layer : Input object
        '''
        if not isinstance(layer, Input):
            raise ValueError

        if self.layers_list:
            if isinstance(self.layers_list[0], Input):
                self.layers_list = self.layers_list[1:]

        self.layers_list.insert(0, layer)

    def add_layer(self, layer, position=None):
        '''
        add_layer
        layer : Layer object
        '''
        if not isinstance(layer, Layer):
            raise ValueError
        if not self.layers_list:
            raise ValueError('Network is empty')

        if position is None:
            self.layers_list.append(layer)
        elif isinstance(position, int):
            self.layers_list.insert(position, layer)
        else:
            raise ValueError(
                'Wrong type of the "position" argument, must be int')

    def add_output(self, layer):
        '''
        add_output
        layer : Layer object
        '''
        if not layer.activ_name == 'sigmoid':
            raise ValueError(
                'Output layer must have Sigmoid activation function')
        if layer.n_units != 1:
            raise ValueError('Output layer must contain only 1 unit')
        self.add_layer(layer)

    def show_network(self):
        '''print_network'''
        print("\nLayers in the network:")
        for idx, item in enumerate(self.layers_list):
            print("   layer {}: {}, units: {}".format(
                idx, item.layer_name, item.n_units))
        print()

    def forward_prop(self):
        '''forward_prop'''
        if not isinstance(self.layers_list[0], Input):
            raise ValueError('Create input layer first')
        if self.layers_list[-1].n_units != 1 \
                or self.layers_list[-1].activ_name != 'sigmoid':
            raise ValueError('Output layer is wrong')
        if len(self.layers_list) == 1:
            raise ValueError('Add more layers first')
        if not self.data:
            raise ValueError('Load data')

        for idx, lay in enumerate(self.layers_list):
            if isinstance(lay, Layer) and idx >= 1:
                lay_prev = self.layers_list[idx - 1]
                lay.linear = np.dot(lay.weight, lay_prev.activation)
                lay.linear += lay.bias
                lay.activate()

        if not np.all(self.layers_list[-1].activation < 1.0):
            plt.figure()
            plt.plot(np.squeeze(self.layers_list[-1].linear))
            plt.show()
            raise ValueError("Value(s) in A[L] == 1, so log(1-A[L]) == -inf")

    def backward_prop(self):
        '''backward_prop'''
        if not isinstance(self.layers_list[-1].activation, np.ndarray):
            raise ValueError('Do forward prop first')

        self.layers_list[-1].d_linear = self.layers_list[-1].activation - \
            self.data.labels

        # https://stackoverflow.com/questions/529424/traverse-a-list-in-reverse-order-in-python
        for idx, lay in reversed(list(enumerate(self.layers_list))):
            if isinstance(lay, Layer) and idx >= 1:
                if not isinstance(lay.activation, np.ndarray):
                    raise ValueError('Do forward prop first')
                lay_prev = self.layers_list[idx - 1]
                lay.d_weight = 1 / self.data.n_examples * \
                    np.dot(lay.d_linear, lay_prev.activation.T)
                lay.d_bias = 1 / self.data.n_examples * \
                    np.sum(lay.d_linear, axis=1, keepdims=True)
                if idx >= 2:
                    lay_prev.d_activation = np.dot(lay.weight.T, lay.d_linear)
                    lay_prev.d_linear = np.multiply(
                        lay_prev.d_activation, lay_prev.derivate())

    def update_weights(self, learn_rate, lambd):
        '''update_weights'''
        for idx, lay in enumerate(self.layers_list):
            if isinstance(lay, Layer) and idx >= 1:
                lay.d_weight += lambd / self.data.n_examples * lay.weight
                lay.weight -= learn_rate * lay.d_weight
                lay.bias -= learn_rate * lay.d_bias

    def calc_cost(self):
        '''calc_cost'''
        if not isinstance(self.layers_list[-1].activation, np.ndarray):
            raise ValueError('Do forward prop first')

        probabilities = self.layers_list[-1].activation
        real_labels = self.data.labels

        val = np.multiply(real_labels, np.log(probabilities)) + \
            np.multiply((1 - real_labels), np.log(1 - probabilities))
        assert val.shape == (real_labels.shape[0], real_labels.shape[1])

        val = np.sum(val, axis=0, keepdims=True)
        assert val.shape == (1, real_labels.shape[1])

        val = -np.mean(val, axis=1, keepdims=True)
        assert val.shape == (1, 1)

        cost = val.item()

        return cost

    def train(self, input_data, epochs, learn_rate, learn_decay, lambd):
        '''train'''
        cost_list = list()

        self.reset_network()
        self.load_data(input_data)

        for epoch in range(epochs + 1):
            try:
                learn_rate_loc = learn_rate / (1 + epoch * learn_decay)

                self.forward_prop()
                cost = self.calc_cost()
                self.backward_prop()
                self.update_weights(learn_rate_loc, lambd)

                if epoch == 0 or epoch % 1000 == 0:
                    cost_list.append(cost)
                    print("Epoch {} cost: {}".format(epoch, cost))

                # if cost_array.shape[0] >= 2:
                #     assert(cost_array[-1, 0] <= cost_array[-2, 0])
            except KeyboardInterrupt:
                break

        return cost_list

    def reset_network(self):
        '''reset all weights and biases'''
        for item in self.layers_list:
            if isinstance(item, Layer):
                item.reset_weights()

    def predict(self, input_data, threshold=0.5):
        '''
        predict
        input_data : Data
        threshold : float, betweeen 0 and 1
        '''
        if not isinstance(input_data, Data):
            raise ValueError('Pass Data object')

        self.load_data(input_data)
        self.forward_prop()
        probabilities = self.layers_list[-1].activation
        predictions = np.array(probabilities >= threshold, dtype=np.int32)

        return predictions

    def predict_and_compare(self, input_data, threshold=0.5):
        '''
        predict_and_compare
        input_data : Data
        threshold : float, betweeen 0 and 1
        '''
        if not isinstance(input_data, Data):
            raise ValueError('Pass Data object')

        predictions = self.predict(input_data, threshold)
        prediction_metrics = self.__calc_prediction_metrics(
            predictions=predictions,
            dataset=input_data
        )
        prediction_metrics[input_data.dataset_name + '_cost'] \
            = self.calc_cost()

        print("\nResults for the \"{}\" dataset".format(input_data.dataset_name))
        for key, value in prediction_metrics.items():
            print(str(key) + " : " + str(value))

        return predictions, prediction_metrics

    @staticmethod
    def __calc_prediction_metrics(predictions, dataset):
        '''
        calc_prediction_metrics
        predictions : numpy.ndarray
        dataset : Data object
        '''
        assert isinstance(dataset.labels, np.ndarray)

        true_positives = np.sum(np.array(
            predictions + dataset.labels == 2,
            dtype=np.int32
        ))

        true_negatives = np.sum(np.array(
            predictions + dataset.labels == 0,
            dtype=np.int32
        ))

        false_positives = np.sum(np.array(
            predictions - dataset.labels == 1,
            dtype=np.int32
        ))

        false_negatives = np.sum(np.array(
            predictions - dataset.labels == -1,
            dtype=np.int32
        ))

        accuracy = (true_positives + true_negatives) / dataset.labels.shape[1]
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1score = 2 * precision * recall / (precision + recall)

        prediction_metrics = {
            dataset.dataset_name + '_accuracy': accuracy,
            dataset.dataset_name + '_precision': precision,
            dataset.dataset_name + '_recall': recall,
            dataset.dataset_name + '_f1score': f1score,
        }

        return prediction_metrics

    def grad_check(self):
        '''
        grad_check
        Check if gradient descent algorithm works correctly
        '''

        def calc_activation_eps(testnet, theta_with_eps):
            '''calc_activation_eps'''
            testnet.layers_list[1].weight = theta_with_eps[:, :-1]  # weights
            testnet.layers_list[1].bias = theta_with_eps[:, [-1]]  # bias

            testnet.forward_prop()
            cost = testnet.calc_cost()

            return cost

        print("\nPerforming gradient check")
        testnet = BinaryClassifierNetwork()
        testnet.add_input(Input(4))
        testnet.add_output(Layer('output', 'sigmoid', 1, 4))
        testnet.show_network()

        examples = np.random.randn(4, 5)
        labels = np.random.randn(1, 5)

        testnet.load_data(Data(
            dataset_name='grad_check',
            features=examples,
            labels=labels
        ))

        testnet.forward_prop()
        testnet.backward_prop()

        weights = testnet.layers_list[1].weight
        biases = testnet.layers_list[1].bias
        d_weights = testnet.layers_list[1].d_weight
        d_biases = testnet.layers_list[1].d_bias

        theta = np.concatenate((weights, biases), axis=1)
        dtheta = np.concatenate((d_weights, d_biases), axis=1)

        eps = 1e-6

        dtheta_approx = np.zeros(theta.shape)

        for idx in np.ndindex(theta.shape):
            theta_eps = np.zeros(theta.shape)
            theta_eps[idx] = eps

            theta_plus = theta + theta_eps
            theta_mnus = theta - theta_eps

            cost_plus = calc_activation_eps(testnet, theta_plus)
            cost_mnus = calc_activation_eps(testnet, theta_mnus)

            dtheta_approx[idx] = (cost_plus - cost_mnus) / (2 * eps)

        error = LA.norm(dtheta_approx - dtheta, 2) / \
            (LA.norm(dtheta_approx, 2) + LA.norm(dtheta, 2))

        if error <= 1e-7:
            print("Grad check is OK")
        else:
            raise Exception("Grad check has failed")
