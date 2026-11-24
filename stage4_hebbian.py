"""
STAGE 4: HEBBIAN LEARNING  ("Neurons that fire together wire together")
-----------------------------------------------------------------------
In stages 2-3 an error signal guided the learning (backpropagation). There is no strong
evidence that the brain has such a backward error signal. A rule closer to learning in
the brain is Hebb's rule:

        delta_w = learning_rate * (presynaptic activity) * (postsynaptic activity)

Two experiments:
  A) Pavlov's dog: after the bell is paired with food, the bell alone makes the dog salivate.
  B) Hopfield memory: store patterns with Hebb's rule and recall them from a corrupted version.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

# =====================================================================
# EXPERIMENT A: PAVLOV
# =====================================================================
print("=" * 60)
print("EXPERIMENT A: Pavlov's dog")
print("=" * 60)

w_food = 1.0    # Food -> salivation: strong innate connection (a reflex)
w_bell = 0.0    # Bell -> salivation: no connection at the start
THRESHOLD = 0.5  # The salivation neuron fires if its total input exceeds this value
ETA = 0.15       # Learning rate
DECAY = 0.10     # A connection that is not reinforced weakens over time (extinction)


def salivation_neuron(bell, food):
    total = bell * w_bell + food * w_food
    return 1 if total > THRESHOLD else 0


print(f"Start: bell connection = {w_bell:.2f}")
print(f"Salivate when only the bell rings? -> {salivation_neuron(1, 0)}  (the dog does not know yet)\n")

history = [w_bell]
print("Training: bell and food are given together")
for trial in range(1, 13):
    bell, food = 1, 1
    salivation = salivation_neuron(bell, food)
    w_bell += ETA * bell * salivation * (1 - w_bell)   # Hebb's rule (saturating at 1)
    history.append(w_bell)
    print(f"  Trial {trial:2d}: bell connection = {w_bell:.2f} | bell only -> salivation {salivation_neuron(1, 0)}")

print("\nExtinction: the bell rings but no food comes (the connection weakens)")
for trial in range(1, 13):
    w_bell *= (1 - DECAY)
    history.append(w_bell)
    if trial % 3 == 0:
        print(f"  Extinction {trial:2d}: bell connection = {w_bell:.2f} | bell only -> salivation {salivation_neuron(1, 0)}")

plt.figure(figsize=(7, 3.5))
plt.plot(history, marker="o")
plt.axhline(THRESHOLD, color="red", linestyle="--", label="salivation threshold")
plt.axvline(12.5, color="gray", linestyle=":")
plt.text(3, 0.05, "learning", fontsize=10)
plt.text(16, 0.05, "extinction", fontsize=10)
plt.xlabel("trial"); plt.ylabel("bell -> salivation connection strength")
plt.title("Pavlov: conditioning with Hebb's rule")
plt.legend(); plt.tight_layout()
plt.savefig("stage4_pavlov.png", dpi=110)

# =====================================================================
# EXPERIMENT B: HOPFIELD ASSOCIATIVE MEMORY
# =====================================================================
print("\n" + "=" * 60)
print("EXPERIMENT B: Hopfield memory (storing patterns with Hebb's rule)")
print("=" * 60)

LETTERS = {
    "X": ["#........#", ".#......#.", "..#....#..", "...#..#...", "....##....",
          "....##....", "...#..#...", "..#....#..", ".#......#.", "#........#"],
    "T": ["##########", "##########", "....##....", "....##....", "....##....",
          "....##....", "....##....", "....##....", "....##....", "....##...."],
    "L": ["##........", "##........", "##........", "##........", "##........",
          "##........", "##........", "##........", "##########", "##########"],
}


def to_pattern(rows):
    # '#' -> +1 (neuron is firing), '.' -> -1 (neuron is silent)
    return np.array([1 if ch == "#" else -1 for row in rows for ch in row])


patterns = {name: to_pattern(rows) for name, rows in LETTERS.items()}
N = 100  # 10x10 = 100 neurons, all connected to each other

# Hebb's rule: the connection between two neurons strengthens if they are in the same state together
W = np.zeros((N, N))
for p in patterns.values():
    W += np.outer(p, p) / N
np.fill_diagonal(W, 0)  # A neuron is not connected to itself
print(f"Stored {len(patterns)} patterns ({', '.join(patterns)}). Number of connections: {N * (N - 1)}")


def recall(start, max_epochs=20):
    state = start.copy()
    for _ in range(max_epochs):
        previous = state.copy()
        for i in rng.permutation(N):             # Update neurons in random order
            state[i] = 1 if W[i] @ state >= 0 else -1
        if np.array_equal(state, previous):      # No more change: the memory has settled
            break
    return state


def corrupt(pattern, fraction):
    noisy = pattern.copy()
    chosen = rng.choice(N, int(N * fraction), replace=False)
    noisy[chosen] *= -1                          # Flip the chosen pixels
    return noisy


fractions = [0.1, 0.2, 0.3, 0.4]
print("\nRecall by corruption level (50 trials per pattern):")
for fraction in fractions:
    successes = 0; total = 0
    for name, p in patterns.items():
        for _ in range(50):
            total += 1
            successes += np.array_equal(recall(corrupt(p, fraction)), p)
    print(f"  {int(fraction * 100)}% corrupted -> perfectly recalled: {successes / total * 100:.0f}%")

# Visual: original / corrupted / recalled
fig, axes = plt.subplots(3, len(patterns), figsize=(2.6 * len(patterns), 7.5))
for k, (name, p) in enumerate(patterns.items()):
    noisy = corrupt(p, 0.25)
    recalled = recall(noisy)
    for row, (image, title) in enumerate([(p, f"Stored: {name}"),
                                          (noisy, "25% corrupted input"),
                                          (recalled, "Recalled")]):
        axes[row, k].imshow(image.reshape(10, 10), cmap="gray_r")
        axes[row, k].set_title(title, fontsize=9)
        axes[row, k].axis("off")
plt.suptitle("Hopfield network: recalls the full memory from a corrupted cue")
plt.tight_layout()
plt.savefig("stage4_hopfield.png", dpi=110)
print("\nImages saved: stage4_pavlov.png, stage4_hopfield.png")
