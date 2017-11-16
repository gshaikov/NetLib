# -*- coding: utf-8 -*-
"""
My custom neural net API

@author: gshai
"""

import numpy as np
import numpy.linalg as LA


#####################################################


class Layer(object):
    '''
    Hidden and output layer
    Activations:
    * relu
    * sigmoid
    * softmax
    '''

    activation_functions = [
        'relu',
        'sigmoid',
        'softmax',
    ]

    # Adam parameters
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    adam_eps = 1e-8

    def __init__(self, layer_name, activ_name, n_units, n_connections):
        if not isinstance(layer_name, str):
            raise ValueError
        elif not activ_name in self.activation_functions:
            raise ValueError
        elif not isinstance(activ_name, str):
            raise ValueError
        elif not (isinstance(n_units, int) or n_units > 0):
            raise ValueError
        elif not (isinstance(n_connections, int) or n_connections > 0):
            raise ValueError

        self.layer_name = layer_name
        self.activ_name = activ_name

        self.n_units = n_units
        self.n_connections = n_connections

        self.weight = None
        self.bias = None

        self.d_weight = None
        self.d_bias = None

        self.linear = None
        self.activation = None

        self.d_linear = None
        self.d_activation = None

        # Adam optimizer variables
        self.vd_weight = None
        self.sd_weight = None
        self.vd_bias = None
        self.sd_bias = None

        self.reset_weights()

    def reset_weights(self):
        '''reset_weights'''
        self.weight = np.random.randn(self.n_units, self.n_connections)
        self.bias = np.zeros((self.n_units, 1))

        if self.activ_name == 'relu':
            self.weight *= np.sqrt(2 / self.n_connections)
        elif self.activ_name in ['sigmoid', 'softmax']:
            self.weight *= np.sqrt(1 / self.n_connections)
        else:
            raise Exception("Activation function not specified")

        self.vd_weight = 0
        self.sd_weight = 0
        self.vd_bias = 0
        self.sd_bias = 0

    def activate(self):
        '''activate'''
        if self.activ_name == 'relu':
            self.activation = self.relu(self.linear)
        elif self.activ_name == 'sigmoid':
            self.activation = self.sigm(self.linear)
        elif self.activ_name == 'softmax':
            self.activation = self.softmax(self.linear)
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
        '''relu'''
        val = lin_z * (lin_z > 0) + 0.01 * lin_z * (lin_z < 0)  # leaky relu
        # https://stackoverflow.com/questions/911871/detect-if-a-numpy-array-contains-at-least-one-non-numeric-value
        assert not np.isnan(val).any()
        return val

    @staticmethod
    def drelu(lin_z):
        '''drelu'''
        val = (lin_z > 0) + 0.01 * (lin_z < 0)  # leaky relu
        assert not np.isnan(val).any()
        return val

    @staticmethod
    def sigm(lin_z):
        '''sigm'''
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html
        lin_z = np.clip(lin_z, -30, 30)  # prevent np.exp() overflow
        val = np.exp(lin_z) / (1 + np.exp(lin_z))
        assert not np.isnan(val).any()
        return val

    def dsigm(self, lin_z):
        '''dsigm'''
        sigmoid_activation = self.sigm(lin_z)
        val = sigmoid_activation * (1 - sigmoid_activation)
        assert not np.isnan(val).any()
        return val

    @staticmethod
    def softmax(lin_z):
        '''softmax'''
        lin_z = np.clip(lin_z, -30, 30)  # prevent np.exp() overflow
        val = np.exp(lin_z) / np.sum(np.exp(lin_z), axis=0, keepdims=True)
        assert not np.isnan(val).any()
        return val


class Input(object):
    '''Input layer'''

    def __init__(self, n_units):
        if not (isinstance(n_units, int) or n_units > 0):
            raise ValueError

        self.layer_name = 'input'
        self.activation = None
        self.n_units = n_units


class DataInNetwork(object):
    '''
    Data
    features and labels are numpy.ndarray type
    * features have dimentions n by m, where
    * * n - number of features
    * * m - number of data examples
    * labels have dimentions 1 by m
    '''

    def __init__(self,
                 dataset_name=None,
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
        self.n_labels = None

        if no_labels is False:
            assert isinstance(labels, np.ndarray)
            assert labels.shape[0] <= labels.shape[1]

            # number of data examples is the same
            assert features.shape[1] == labels.shape[1]

            self.labels = labels
            self.n_labels = labels.shape[0]

    def trimmed(self, size):
        '''trim the dataset so it becomes shorter'''
        new_features = self.features[:, :size]
        if self.labels is None:
            data_trimmed = DataInNetwork(
                dataset_name='trimmed',
                features=new_features,
                no_labels=True
            )
        else:
            new_labels = self.labels[:, :size]
            data_trimmed = DataInNetwork(
                dataset_name='trimmed',
                features=new_features,
                labels=new_labels
            )
        return data_trimmed


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
        if not isinstance(input_data, DataInNetwork):
            raise ValueError('Pass Data object')
        elif not isinstance(self.layers_list[0], Input):
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
                del self.layers_list[0]

        self.layers_list.insert(0, layer)

    def add_layer(self, layer, position=None):
        '''
        add_layer
        layer : Layer object
        '''
        if not isinstance(layer, Layer):
            raise ValueError
        elif not self.layers_list:
            raise ValueError('Network is empty')
        elif layer.n_connections != self.layers_list[-1].n_units:
            raise ValueError(
                'layer.n_connections != self.layers_list[-1].n_units')
        elif not isinstance(position, int) and not position is None:
            raise ValueError('Position argument must be int')

        if position is None:
            self.layers_list.append(layer)
        else:
            self.layers_list.insert(position, layer)

    def add_output(self, layer):
        '''
        add_output
        layer : Layer object
        '''
        if not layer.activ_name in ['sigmoid', 'softmax']:
            raise ValueError('Wrong activation function')

        self.add_layer(layer)

    def show_network(self):
        '''print_network'''
        print("\nLayers in the network:")
        for idx, item in enumerate(self.layers_list):
            print("   layer {}: {}, units: {}".format(
                idx, item.layer_name, item.n_units))
        print()

    def reset_network(self):
        '''reset all weights and biases'''
        for item in self.layers_list:
            if isinstance(item, Layer):
                item.reset_weights()

    def forward_prop(self):
        '''forward_prop'''
        if not isinstance(self.layers_list[0], Input):
            raise ValueError('Create input layer first')
        elif not self.layers_list[-1].activ_name in ['sigmoid', 'softmax']:
            raise ValueError('Output layer is wrong')
        elif not self.data:
            raise ValueError('Load data')

        for idx, lay in enumerate(self.layers_list):
            if isinstance(lay, Layer) and idx >= 1:
                lay_prev = self.layers_list[idx - 1]
                lay.linear = np.dot(lay.weight, lay_prev.activation)
                lay.linear += lay.bias
                lay.activate()

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

                lay.d_weight = 1 / self.data.n_examples \
                    * np.dot(lay.d_linear, lay_prev.activation.T)
                lay.vd_weight = lay.adam_beta_1 * lay.vd_weight \
                    + (1 - lay.adam_beta_1) * lay.d_weight
                lay.sd_weight = lay.adam_beta_2 * lay.sd_weight \
                    + (1 - lay.adam_beta_2) * (lay.d_weight ** 2)

                lay.d_bias = 1 / self.data.n_examples * \
                    np.sum(lay.d_linear, axis=1, keepdims=True)
                lay.vd_bias = lay.adam_beta_1 * lay.vd_bias \
                    + (1 - lay.adam_beta_1) * lay.d_bias
                lay.sd_bias = lay.adam_beta_2 * lay.sd_bias \
                    + (1 - lay.adam_beta_2) * (lay.d_bias ** 2)

                if idx >= 2:
                    lay_prev.d_activation = np.dot(lay.weight.T, lay.d_linear)

                    lay_prev.d_linear = np.multiply(
                        lay_prev.d_activation, lay_prev.derivate())

    def update_weights(self, learn_rate, lambd):
        '''update_weights'''
        for idx, lay in enumerate(self.layers_list):
            if isinstance(lay, Layer) and idx >= 1:
                regularizer = lambd / self.data.n_examples * lay.weight
                # update_weight = regularizer + lay.d_weight
                # update_bias = lay.d_bias
                update_weight = regularizer \
                    + lay.vd_weight / (np.sqrt(lay.sd_weight) + lay.adam_eps)
                update_bias \
                    = lay.vd_bias / (np.sqrt(lay.sd_bias) + lay.adam_eps)
                lay.weight = lay.weight - learn_rate * update_weight
                lay.bias = lay.bias - learn_rate * update_bias

    def calc_cost(self):
        '''calc_cost'''
        if not isinstance(self.layers_list[-1].activation, np.ndarray):
            raise ValueError('Do forward prop first')

        probabilities = self.layers_list[-1].activation
        real_labels = self.data.labels

        # prevent np.log(0) -> -inf
        probabilities -= 1e-10 * (probabilities == 1)
        probabilities += 1e-10 * (probabilities == 0)

        val = np.multiply(real_labels, np.log(probabilities))
        if self.layers_list[-1].activ_name == 'sigmoid':
            val += np.multiply((1 - real_labels), np.log(1 - probabilities))
        assert val.shape == (real_labels.shape[0], real_labels.shape[1])
        val = np.sum(val, axis=0, keepdims=True)
        assert val.shape == (1, real_labels.shape[1])
        val = -np.mean(val, axis=1, keepdims=True)
        assert val.shape == (1, 1)
        cost = val.item()
        return cost

    def train(self, learn_rate, lambd):
        '''train'''
        self.forward_prop()
        cost = self.calc_cost()
        self.backward_prop()
        self.update_weights(learn_rate, lambd)
        # if cost_array.shape[0] >= 2:
        #     assert(cost_array[-1, 0] <= cost_array[-2, 0])
        return cost

    def predict(self, input_data):
        '''
        predict
        input_data : Data
        threshold : float, betweeen 0 and 1
        '''
        if not isinstance(input_data, DataInNetwork):
            raise ValueError('Pass Data object')

        self.load_data(input_data)
        self.forward_prop()
        probabilities = self.layers_list[-1].activation
        predictions = self.softmax_to_category(probabilities)
        return predictions

    @staticmethod
    def softmax_to_category(probabilities):
        '''
        convert from softmax vector output to int
        https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argmax.html
        '''
        return np.argmax(probabilities, axis=0).reshape(1, probabilities.shape[1])

    def predict_and_compare(self, input_data):
        '''
        predict_and_compare
        input_data : Data
        threshold : float, betweeen 0 and 1
        '''
        if not isinstance(input_data, DataInNetwork):
            raise ValueError('Pass Data object')

        predictions = self.predict(input_data)
        prediction_metrics = self.__calc_prediction_metrics(
            predictions=predictions,
            dataset=input_data
        )
        prediction_metrics['cost'] = self.calc_cost()
        return predictions, prediction_metrics

    def __calc_prediction_metrics(self, predictions, dataset):
        '''
        calc_prediction_metrics
        predictions : numpy.ndarray
        dataset : Data object
        '''
        assert isinstance(dataset.labels, np.ndarray)

        correct_labels = self.softmax_to_category(dataset.labels)
        total_correct = np.sum(
            predictions == correct_labels,
            axis=1,
            keepdims=True,
        )
        accuracy = float(total_correct / predictions.shape[1])
        prediction_metrics = {
            'accuracy': accuracy,
        }
        return prediction_metrics

    @classmethod
    def grad_check(cls):
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
        testnet = cls()
        testnet.add_input(Input(4))
        testnet.add_output(Layer('output', 'sigmoid', 1, 4))
        testnet.show_network()

        examples = np.random.randn(4, 5)
        labels = np.random.randn(1, 5)

        testnet.load_data(DataInNetwork(
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
