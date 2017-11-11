# NetAPI

## Practicing Neural Networks

This is my binary classifier network API, for fun.

Tested on a classic hand-written digit dataset.

## Structure
- ``main.py`` is the main file
- ``myneuralnetapi.py`` is the neural network API module

## Test
Layers in the network:
layer 0: input, units: 2
layer 1: hidden, units: 10
layer 2: hidden, units: 10
layer 3: hidden, units: 10
layer 4: hidden, units: 10
layer 5: output, units: 4

epochs = 30000

learn_rate = 0.01
learn_decay = 1 / 5000 * 0

lambd = 1.0 * 1
threshold = 0.5

