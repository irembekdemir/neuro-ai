"""
STAGE 3: HANDWRITTEN DIGIT RECOGNITION (a neural network from scratch with NumPy)
--------------------------------------------------------------------------------
Data:    1797 handwritten digits bundled with scikit-learn (8x8 pixels = 64 inputs).
Network: 64 inputs -> 32 hidden neurons -> 10 outputs (one neuron per digit, 0 to 9)

The same logic as stage 2, but with NumPy matrices: instead of looping over neurons
one by one, we compute a whole layer in a single operation.
(No neural network library is used; scikit-learn only provides the dataset and
matplotlib is only used for plotting.)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Draw to a file, no screen needed
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

# --- 1) Prepare the data ---
digits = load_digits()
X = digits.data / 16.0           # Pixel values are 0-16, scale them to 0-1
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def one_hot(labels, num_classes=10):
    # 3 -> [0,0,0,1,0,0,0,0,0,0]  : "the correct neuron is 1, all others are 0"
    m = np.zeros((len(labels), num_classes))
    m[np.arange(len(labels)), labels] = 1
    return m


Y_train = one_hot(y_train)

# --- 2) Weights (connection strengths) ---
HIDDEN = 32
W1 = rng.normal(0, 0.3, (64, HIDDEN)); b1 = np.zeros(HIDDEN)
W2 = rng.normal(0, 0.3, (HIDDEN, 10)); b2 = np.zeros(10)


def forward(X):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)                      # ReLU: silence negative values, pass positive ones
    z2 = a1 @ W2 + b2
    e = np.exp(z2 - z2.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)         # Softmax: turns the 10 outputs into probabilities
    return a1, p


def accuracy(X, y):
    return (forward(X)[1].argmax(axis=1) == y).mean()


# --- 3) Training ---
learning_rate = 0.5
num_epochs = 300
print(f"Training starts: {len(X_train)} training images, {len(X_test)} images held out for testing\n")
print(f"Initial test accuracy (random network): {accuracy(X_test, y_test) * 100:.1f}%")

for epoch in range(1, num_epochs + 1):
    a1, p = forward(X_train)
    # Backpropagation (for softmax + cross-entropy the error signal is prediction - truth)
    d2 = (p - Y_train) / len(X_train)
    dW2 = a1.T @ d2; db2 = d2.sum(axis=0)
    d1 = (d2 @ W2.T) * (a1 > 0)
    dW1 = X_train.T @ d1; db1 = d1.sum(axis=0)
    W2 -= learning_rate * dW2; b2 -= learning_rate * db2
    W1 -= learning_rate * dW1; b1 -= learning_rate * db1
    if epoch in (1, 10, 50, 100, 200, 300):
        print(f"Epoch {epoch:3d}: train accuracy {accuracy(X_train, y_train) * 100:5.1f}% | "
              f"test accuracy {accuracy(X_test, y_test) * 100:5.1f}%")

final_accuracy = accuracy(X_test, y_test)
print(f"\nFINAL: accuracy on {len(X_test)} images the network has never seen = {final_accuracy * 100:.1f}%")

# --- 4) Noise test: the brain can recognize degraded images too, how about our network? ---
print("\nNoise test (adding random distortion to the pixels):")
for level in (0.0, 0.1, 0.2, 0.3, 0.5):
    noisy = np.clip(X_test + rng.normal(0, level, X_test.shape), 0, 1)
    print(f"  noise {level:.1f} -> accuracy {accuracy(noisy, y_test) * 100:.1f}%")

# --- 5) Visual outputs ---
predictions = forward(X_test)[1].argmax(axis=1)
fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
    is_correct = predictions[i] == y_test[i]
    ax.set_title(f"Predicted: {predictions[i]}", color="green" if is_correct else "red", fontsize=9)
    ax.axis("off")
plt.suptitle(f"Predictions on test images (accuracy {final_accuracy * 100:.1f}%)")
plt.tight_layout()
plt.savefig("stage3_predictions.png", dpi=110)

fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(W1[:, i].reshape(8, 8), cmap="RdBu")
    ax.axis("off")
plt.suptitle("Patterns the hidden neurons 'look at' (blue: excitatory, red: inhibitory)")
plt.tight_layout()
plt.savefig("stage3_neuron_patterns.png", dpi=110)
print("\nImages saved: stage3_predictions.png, stage3_neuron_patterns.png")
