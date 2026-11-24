# Brain Project: A Step-by-Step Introduction to Neural Networks

> **Status: work in progress.** This is a learning project that is still being developed. The code and explanations will change over time, and there may be gaps or mistakes.

A project that tries to model the basic ideas of the human brain (neurons, connections, learning) with small code examples. Each stage builds on the previous one, going from simple to more complex. The brain itself cannot be coded one-to-one; everything here is a heavily simplified model.

## What has been done so far

| Stage | File | Topic | Required libraries |
|-------|------|-------|--------------------|
| 1 | `stage1_perceptron.py` | A single neuron (perceptron). Learns the AND and OR gates, cannot learn XOR. | None (pure Python) |
| 2 | `stage2_xor_network.py` | A small neural network with a hidden layer; solves XOR with backpropagation. | None (pure Python) |
| 3 | `stage3_digit_recognition.py` | A neural network written from scratch with NumPy, recognizing handwritten digits. | numpy, scikit-learn, matplotlib |
| 4 | `stage4_hebbian.py` | Hebbian learning: Pavlov's dog and a Hopfield memory. | numpy, matplotlib |

### Notes on each stage

- **Stage 1:** Shows the ideas of weights, bias and threshold. At the end it tries XOR to show why a single neuron cannot solve it (it is not linearly separable).
- **Stage 2:** A 2-3-1 network with sigmoid activation and backpropagation.
- **Stage 3:** A 64-32-10 network trained on the 8x8 pixel digits dataset bundled with scikit-learn. Test accuracy is around 97%. No neural network library is used: scikit-learn only provides the dataset and matplotlib is only used for plotting. It also runs a noise test and produces image outputs.
- **Stage 4:** Instead of backpropagation, it uses the rule "neurons that fire together wire together" (Hebb's rule). It demonstrates Pavlov's dog (conditioning and extinction) and a Hopfield network (recalling a pattern from a corrupted cue).

## Setup and usage

Python 3 is required. For stages 3 and 4:

```bash
pip install numpy scikit-learn matplotlib
```

Run the files in order:

```bash
python stage1_perceptron.py
python stage2_xor_network.py
python stage3_digit_recognition.py
python stage4_hebbian.py
```

Stages 3 and 4 save image files (`.png`) to the current folder when they run:

- `stage3_predictions.png`: test images and the network's predictions
- `stage3_neuron_patterns.png`: the patterns the hidden neurons "look at"
- `stage4_pavlov.png`: conditioning and extinction plot
- `stage4_hopfield.png`: stored, corrupted and recalled patterns

## Limitations

- Real neurons work in time (they send spikes). These models work with instantaneous numbers.
- There is no strong evidence that backpropagation works in the brain in this form. Stage 4 explores a rule that is closer to biology.
- The networks are tiny: tens to hundreds of neurons. The human brain has about 86 billion.

## To do / ideas

- [ ] Change the number of hidden neurons in stage 3 and compare the effect on accuracy
- [ ] Increase the number of stored patterns in stage 4 and see when the Hopfield memory starts to break down
- [ ] Add a time-based neuron model (Leaky Integrate-and-Fire)
- [ ] Move the code into a cleaner structure (separate the shared parts)
- [ ] Maybe write some of the stages in C++ as well

## Project structure

```
.
├── stage1_perceptron.py
├── stage2_xor_network.py
├── stage3_digit_recognition.py
├── stage4_hebbian.py
└── README.md
```


