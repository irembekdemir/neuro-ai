"""
STAGE 1: A SINGLE NEURON (PERCEPTRON)
-------------------------------------
Biological neuron:  dendrites (inputs) -> cell body (summation) -> axon (output)
Our neuron:         inputs * weights -> sum + bias -> threshold decision (0 or 1)

Learning = nudging the weights a little every time the neuron gets an answer wrong.
"""
import random

random.seed(1)  # Same result on every run


class Neuron:
    def __init__(self, num_inputs):
        # Connection strengths start out random: the neuron knows nothing yet
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)  # How "eager" the neuron is to fire

    def fire(self, inputs):
        total = self.bias
        for x, w in zip(inputs, self.weights):
            total += x * w
        # Threshold: fire (1) if the total is above 0, stay silent (0) otherwise
        return 1 if total > 0 else 0

    def learn(self, inputs, target, learning_rate=0.1):
        error = target - self.fire(inputs)  # -1, 0 or +1
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * error * inputs[i]
        self.bias += learning_rate * error
        return abs(error)


def train(name, data, num_epochs=50):
    print(f"\n=== Learning the {name} gate ===")
    neuron = Neuron(2)
    print(f"Initial weights: {[round(w, 2) for w in neuron.weights]}, bias: {round(neuron.bias, 2)}")
    for epoch in range(1, num_epochs + 1):
        total_errors = sum(neuron.learn(x, t) for x, t in data)
        if epoch <= 3 or total_errors == 0:
            print(f"Epoch {epoch:2d}: errors = {total_errors}, "
                  f"weights = {[round(w, 2) for w in neuron.weights]}, bias = {round(neuron.bias, 2)}")
        if total_errors == 0:
            print(f"-> All answers correct at epoch {epoch}, learning finished.")
            break
    for x, t in data:
        print(f"  {x} -> {neuron.fire(x)} (expected {t})")
    return neuron


AND = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
OR = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
XOR = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

if __name__ == "__main__":
    train("AND", AND)
    train("OR", OR)

    # XOR: a single neuron cannot learn it, because it is not linearly separable.
    print("\n=== Trying XOR (it will fail) ===")
    neuron = Neuron(2)
    for _ in range(1000):
        errors = sum(neuron.learn(x, t) for x, t in XOR)
    print(f"After 1000 epochs there are still {errors} errors. A single neuron cannot solve XOR.")
    print("The fix: add a hidden layer (see stage 2).")
