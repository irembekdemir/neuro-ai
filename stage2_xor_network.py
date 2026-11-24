"""
STAGE 2: A SMALL NEURAL NETWORK THAT SOLVES XOR (2 inputs -> 3 hidden neurons -> 1 output)
------------------------------------------------------------------------------------------
What is new compared to stage 1:
  1) Hidden layer: neurons are now arranged in layers, like the layers of the cortex.
  2) Sigmoid activation: a smooth transition (a value between 0 and 1) instead of a hard 0/1 threshold.
  3) Backpropagation: the error is passed backwards from the output, and every connection
     is corrected in proportion to its share of the error.
"""
import math
import random

random.seed(7)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Layer:
    def __init__(self, num_inputs, num_neurons):
        # weights[j][i] : strength of the connection from input i to neuron j
        self.weights = [[random.uniform(-1, 1) for _ in range(num_inputs)]
                        for _ in range(num_neurons)]
        self.biases = [random.uniform(-1, 1) for _ in range(num_neurons)]
        self.inputs = []
        self.outputs = []

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = []
        for w, b in zip(self.weights, self.biases):
            total = b + sum(x * wi for x, wi in zip(inputs, w))
            self.outputs.append(sigmoid(total))
        return self.outputs

    def backward(self, error_signal, learning_rate):
        """error_signal: the error at this layer's outputs. Returns the signal to pass to the previous layer."""
        previous_error = [0.0] * len(self.inputs)
        for j in range(len(self.weights)):
            # Derivative of the sigmoid: output * (1 - output). How much the neuron is able to change.
            delta = error_signal[j] * self.outputs[j] * (1 - self.outputs[j])
            for i in range(len(self.inputs)):
                previous_error[i] += delta * self.weights[j][i]
                self.weights[j][i] += learning_rate * delta * self.inputs[i]
            self.biases[j] += learning_rate * delta
        return previous_error


class NeuralNetwork:
    def __init__(self, sizes):
        # sizes = [2, 3, 1] -> 2 inputs, 3 hidden neurons, 1 output
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def learn(self, inputs, target, learning_rate=0.5):
        outputs = self.predict(inputs)
        error = [t - o for t, o in zip(target, outputs)]
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)
        return sum((t - o) ** 2 for t, o in zip(target, outputs))


XOR = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

if __name__ == "__main__":
    network = NeuralNetwork([2, 3, 1])
    print("BEFORE training (the network is still random):")
    for x, t in XOR:
        print(f"  {x} -> {network.predict(x)[0]:.3f} (expected {t[0]})")

    print("\nTraining starts...")
    for epoch in range(1, 10001):
        total_error = sum(network.learn(x, t) for x, t in XOR)
        if epoch in (1, 100, 500, 1000, 2000, 5000, 10000):
            print(f"  Epoch {epoch:5d}: total error = {total_error:.5f}")

    print("\nAFTER training:")
    for x, t in XOR:
        output = network.predict(x)[0]
        print(f"  {x} -> {output:.3f}  (rounded: {round(output)}, expected {t[0]})")
    print("\nXOR, which a single neuron cannot learn, is solved thanks to the hidden layer.")
