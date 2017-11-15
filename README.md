# NetAPI

## Practicing Neural Networks

This is my Neural Network API built only using Numpy, for practice.

Tested on a classic hand-written digit dataset.

## Structure

- ``main.py`` is the main file
- ``myneuralnetapi.py`` is the neural network API module

## Testing the network

Testing the network using ```network_test.py``` with dummy data with 2 features for easy visualization. Plots below illustrate that the network does what it's supposed to do. Accuracy is 99.9%, due to overlapping classes.

![blobs](test_results/data.png)
![learned boundaries](test_results/boundaries.png)
![cost per iteration](test_results/cost.png)

> **Training Parameters**
> - epochs = 10000
> - learn_rate = 0.001
> - lambd_val = 300.0

> **Layers in the network**
> - layer 0: input, units: 2
> - layer 1: relu, units: 10
> - layer 2: relu, units: 10
> - layer 3: relu, units: 10
> - layer 4: relu, units: 10
> - layer 5: softmax, units: 4
