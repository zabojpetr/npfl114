#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(len(self._images))
            return True
        return False


class Network:
    WIDTH, HEIGHT = 224, 224
    LABELS = 250

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name="im")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Create NASNet
            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)
            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                features, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=None, is_training=True)

            # self.classifier_saver = tf.train.import_meta_graph('my_classifier-17000.meta')

            self.nasnet_saver = tf.train.Saver()

            # with tf.name_scope("cl"):
            #     classifier = tf.train.import_meta_graph('my_classifier-17000.meta')
            #
            # # Strip off the "net2/" prefix to get the names of the variables in the checkpoint.
            # classifier_saver_varlist = {v.name.lstrip("cl/"): v
            #                 for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="cl/")}
            # self.classifier_saver = tf.train.Saver(var_list=classifier_saver_varlist)
            #
            # graph = tf.get_default_graph()
            # output_layer = graph.get_tensor_by_name("cl/output_layer:0")


            dense = tf.layers.Dense(2000, tf.nn.relu, name="cl/Dense_1")

            output = tf.layers.Dense(self.LABELS, activation=None, name="cl/output_layer")

            # self.classifier_saver = tf.train.import_meta_graph('my_classifier-340.meta')
            # self.classifier_saver.restore(self.session, tf.train.latest_checkpoint('./'))

            # with tf.name_scope("cl"):
            #     cl = tf.train.import_meta_graph('my_classifier-340.meta')
            #
            # # Strip off the "net1/" prefix to get the names of the variables in the checkpoint.
            # cl_varlist = {v.name.lstrip("cl/"): v
            #                 for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="cl/")}
            # self.classifier_saver = tf.train.Saver(var_list=cl_varlist)
            #
            # graph = tf.get_default_graph()
            # dense = graph.get_tensor_by_name("cl/Dense_1:0")
            # output = graph.get_tensor_by_name("cl/output_layer:0")

            output_layer = output(dense(features))

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            self.predictions = tf.argmax(output_layer, 1)
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer)

            global_step = tf.train.create_global_step()

            decay_steps = args.num_examples // args.batch_size * args.epochs / (args.learning_rate_steps + 1)
            decay_rate = np.power(args.learning_rate_finish / args.learning_rate_start, 1 / (args.learning_rate_steps))
            learning_rate = tf.train.exponential_decay(args.learning_rate_start, global_step,
                                                       decay_steps,
                                                       decay_rate, staircase=True)

            self.training = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step,
                                                                           name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/learning_rate", learning_rate),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
                self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            # Load NASNet
            self.nasnet_saver.restore(self.session, tf.train.latest_checkpoint("./finetuning/nasnet3/"))
            # self.nasnet_saver.restore(self.session, args.nasnet)
            self.classifier_saver = tf.train.Saver(dense.variables + output.variables)
            self.classifier_saver.restore(self.session, tf.train.latest_checkpoint('./finetuning/classifier3/'))
            # self.classifier_saver.restore(self.session, tf.train.latest_checkpoint('./'))





    def train_batch(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy], {self.images: batch_images, self.labels: batch_labels, self.is_training: False})
            loss += batch_loss * len(batch_images) / len(dataset.images)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.images)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            images, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.images: images, self.is_training: False}))
        return np.concatenate(labels)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=25, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--nasnet", default="nets/nasnet/model.ckpt", type=str, help="NASNet checkpoint path.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate_start", default=0.00001, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_finish", default=1e-30, type=float, help="Learning rate decreasing.")
    parser.add_argument("--learning_rate_steps", default=25, type=int, help="How much steps to final learning rate")
    args = parser.parse_args()

    if args.learning_rate_steps == -1:
        args.learning_rate_steps = args.epochs

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("nsketch-train.npz")
    dev = Dataset("nsketch-dev.npz", shuffle_batches=False)
    test = Dataset("nsketch-test.npz", shuffle_batches=False)

    args.num_examples = train.images.shape[0]

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels = train.next_batch(args.batch_size)
            network.train_batch(images, labels)

        network.evaluate("dev", dev, args.batch_size)

        network.nasnet_saver.save(network.session, "./finetuning/nasnet4/nasner")
        network.classifier_saver.save(network.session, "./finetuning/classifier4/classifier")

        # Predict test data
        with open("{}/nsketch_transfer_test_{}.txt".format(args.logdir, str(i)), "w") as test_file:
            labels = network.predict(test, args.batch_size)
            for label in labels:
                print(label, file=test_file)
