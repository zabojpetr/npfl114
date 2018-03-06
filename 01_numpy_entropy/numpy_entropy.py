#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    lines = []
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line
            lines.append(line)

    # TODO: Create a NumPy array containing the data distribution
    labels, counts = np.unique(lines, return_counts=True)
    counts = counts/len(lines)
    occurences = dict(zip(labels, counts))


    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    model_occurences = np.zeros(counts.shape)
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line
            split = line.split("\t")
            if np.where(labels==split[0]):
                model_occurences[np.where(labels==split[0])] = float(split[1])

    # TODO: Compute and print entropy H(data distribution)
    entropy = 0
    for value in counts:
        entropy -= value * np.log(value)
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_entropy = 0
    kl_divergence = 0
    for idx in range(np.prod(counts.shape)):
        cross_entropy -= counts[idx] * np.log(model_occurences[idx])
        kl_divergence += counts[idx] * (np.log(counts[idx]) - np.log(model_occurences[idx]))

    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(kl_divergence))
