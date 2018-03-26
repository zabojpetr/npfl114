#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
from distributed import fire_and_forget


class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        # Load the data
        with open(filename, "r", encoding="utf-8") as file:
            self._text = file.read()

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet: break

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text), np.bool)
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in alphabet_map: char = "<unk>"
            self._lcletters[i + window] = alphabet_map[char]
            self._labels[i] = self._text[i].isupper()

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._text))

    def _create_batch(self, permutation):
        batch_windows = np.zeros([len(permutation), 2 * self._window + 1], np.int32)
        for i in range(0, 2 * self._window + 1):
            batch_windows[:, i] = self._lcletters[permutation + i]

        batch_windows_onehot = np.zeros([len(permutation), (2*self._window + 1) * len(self._alphabet)])

        for i in range(0, 2 * self._window + 1):
            batch_windows_onehot[:,i*(len(self._alphabet)):(i+1)*(len(self._alphabet))] = \
                np.apply_along_axis(lambda x: np.bincount(x, minlength=len(self._alphabet)), axis=1, arr=batch_windows[:,i].reshape(-1,1))
        return batch_windows_onehot, self._labels[permutation]

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))
        # self.session = tf.Session(graph = graph)

    def construct(self, alphabet, args):
        with self.session.graph.as_default():
            # Inputs
            self.windows = tf.placeholder(tf.float32, [None, (2 * args.window + 1)*alphabet], name="windows")
            self.labels = tf.placeholder(tf.int64, [None], name="labels") # Or you can use tf.int32
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Define a suitable network with appropriate loss function
            first_layer = tf.layers.dense(self.windows, 1000, activation=tf.nn.relu, name="first_layer")
            second_layer = tf.layers.dense(first_layer, 500, activation=tf.nn.relu, name="second_layer")
            second_layer_dropout = tf.layers.dropout(second_layer, training=self.is_training,
                                                     name="second_layer_dropout")
            output_layer = tf.layers.dense(second_layer_dropout, 2, activation=None, name="output_layer")

            # self.predictions = tf.cast(tf.cast(output_layer, tf.int32), tf.bool)
            self.predictions = tf.argmax(output_layer, axis=1)

            print(self.predictions.shape)

            # TODO: Define training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer()
            self.training = optimizer.minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, windows, labels):
        self.session.run([self.training, self.summaries["train"]], {self.windows: windows, self.labels: labels, self.is_training:True})

    def evaluate(self, dataset, windows, labels):
        return self.session.run([self.predictions, self.summaries[dataset]], {self.windows: windows, self.labels: labels, self.is_training:False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet_size", default=100, type=int, help="Alphabet size.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=20, type=int, help="Size of the window to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("uppercase_data_train.txt", args.window, alphabet=args.alphabet_size)
    dev = Dataset("uppercase_data_dev.txt", args.window, alphabet=train.alphabet)
    test = Dataset("uppercase_data_test.txt", args.window, alphabet=train.alphabet)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(len(train.alphabet), args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            windows, labels = train.next_batch(args.batch_size)
            network.train(windows, labels)

        dev_windows, dev_labels = dev.all_data()
        network.evaluate("dev", dev_windows, dev_labels)

        test_windows, test_labels = test.all_data()
        prediction, _  = network.evaluate("test", test_windows, test_labels)

        print(prediction)
        prediction[0] = 1
        with open("uppercase_data_test.txt", "r", encoding="utf-8") as raw_file:
            with open("test70.txt", "w", encoding="utf-8") as transform_file:
                raw = raw_file.read()

                for i in range(len(raw)):
                    transform_file.write(raw[i] if prediction[i]==0 else raw[i].upper())
                transform_file.flush()
                transform_file.close()

    # TODO: Generate the uppercased test set