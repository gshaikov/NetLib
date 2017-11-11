# -*- coding: utf-8 -*-
"""
My custom neural net API

@author: gshai
"""

import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt


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

        self.weight = None
        self.bias = None

        self.d_weight = None
        self.d_bias = None

        self.linear = None
        self.activation = None

        self.d_linear = None
        self.d_activation = None

        self.layer_name = layer_name
        self.activ_name = activ_name

        self.n_units = n_units
        self.n_connections = n_connections

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
        return lin_z * (lin_z > 0)

    @staticmethod
    def drelu(lin_z):
        '''drelu'''
        return np.array(lin_z > 0, dtype=np.float32)

    def sigm(self, lin_z):
        '''sigm'''
        lin_z = self.limit_z(lin_z)
        return np.exp(lin_z) / (1 + np.exp(lin_z))

    def dsigm(self, lin_z):
        '''dsigm'''
        sigmoid_activation = self.sigm(lin_z)
        return sigmoid_activation * (1 - sigmoid_activation)

    def softmax(self, lin_z):
        '''softmax'''
        # lin_z = self.limit_z(lin_z)
        return np.exp(lin_z) / np.sum(np.exp(lin_z), axis=0, keepdims=True)

    @staticmethod
    def limit_z(lin_z):
        '''limit_z'''
        # apply function to elements in the array
        # https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.vectorize.html
        v_upper = np.vectorize(lambda x: 20 if x > 20 else x)
        v_lower = np.vectorize(lambda x: -20 if x < -20 else x)
        lin_z = v_upper(lin_z)
        lin_z = v_lower(lin_z)
        return lin_z


class Input(object):
    '''Input layer'''

    def __init__(self, n_units):
        if not (isinstance(n_units, int) or n_units > 0):
            raise ValueError

        self.layer_name = 'input'
        self.activation = None
        self.n_units = n_units


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

        true_positives = np.sum(predictions + dataset.labels == 2, axis=1)
        true_negatives = np.sum(predictions + dataset.labels == 0, axis=1)
        false_positives = np.sum(predictions - dataset.labels == 1, axis=1)
        false_negatives = np.sum(predictions - dataset.labels == -1, axis=1)

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
