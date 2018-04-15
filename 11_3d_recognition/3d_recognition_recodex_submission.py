# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._voxels = data[\"voxels\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        for dataset in [first, second]:
            dataset._shuffle_batches = self._shuffle_batches
            dataset._new_permutation()

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._voxels[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False


class Network:
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.voxels = tf.placeholder(
                tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name=\"voxels\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            layers = [self.voxels]
            for layer in args.cnn.split(','):
                items = layer.split('-')
                if items[0] == 'CBT':
                    layers.append(tf.layers.conv3d_transpose(layers[-1], int(items[1]), (int(items[2]), int(items[2]), int(items[2])),
                                                   (int(items[3]), int(items[3]), int(items[3])), items[4], use_bias=False,
                                                   name=\"Conv3D_Transpose_\" + str(len(layers))))
                    layers.append(tf.nn.relu(tf.layers.batch_normalization(layers[-1], training=self.is_training,
                                                                           name=\"BatchNorm_\" + str(len(layers)))))

                if items[0] == 'CB':
                    layers.append(tf.layers.conv3d(layers[-1], int(items[1]), (int(items[2]), int(items[2]), int(items[2])),
                                                   (int(items[3]), int(items[3]), int(items[3])), items[4], use_bias=False,
                                                   name=\"Conv3D_\" + str(len(layers))))
                    layers.append(tf.nn.relu(tf.layers.batch_normalization(layers[-1], training=self.is_training,
                                                                           name=\"BatchNorm_\" + str(len(layers)))))

                elif items[0] == 'CT':
                    layers.append(tf.layers.conv3d_transpose(layers[-1], int(items[1]), (int(items[2]), int(items[2]), int(items[2])),
                                         (int(items[3]), int(items[3]), int(items[3])), items[4], use_bias=False,
                                         name=\"Conv3D_Transpose_\" + str(len(layers))))

                elif items[0] == 'C':
                    layers.append(tf.layers.conv3d(layers[-1], int(items[1]), (int(items[2]), int(items[2]), int(items[2])),
                                         (int(items[3]), int(items[3]), int(items[3])), items[4], use_bias=False,
                                         name=\"Conv3D_\" + str(len(layers))))

                elif items[0] == 'M':
                    layers.append(tf.layers.max_pooling3d(layers[-1], (int(items[1]), int(items[1]), int(items[1])),
                                                          (int(items[2]), int(items[2]), int(items[2])),
                                                          name=\"MaxPool_\" + str(len(layers))))

                elif items[0] == 'F':
                    layers.append(tf.layers.flatten(layers[-1], name=\"Flatten_\" + str(len(layers))))

                elif items[0] == 'R':
                    layers.append(
                        tf.layers.dense(layers[-1], int(items[1]), tf.nn.relu, name=\"Dense_\" + str(len(layers))))

                elif items[0] == 'RD':
                    layers.append(
                        tf.layers.dense(layers[-1], int(items[1]), tf.nn.relu, name=\"Dense_\" + str(len(layers))))
                    layers.append(
                        tf.layers.dropout(layers[-1], training=self.is_training, name=\"Dropout_\" + str(len(layers))))

            output_layer = tf.layers.dense(layers[-1], self.LABELS, activation=tf.nn.relu, name='output_layer')

            self.predictions = tf.argmax(output_layer, axis=1)

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer)
            self.loss = loss
            global_step = tf.train.create_global_step()  # batches to one epoch * epochs/steps
            decay_steps = args.num_examples // args.batch_size * args.epochs / (args.learning_rate_steps + 1)
            decay_rate = np.power(args.learning_rate_finish / args.learning_rate_start, 1 / (args.learning_rate_steps))
            learning_rate = tf.train.exponential_decay(args.learning_rate_start, global_step,
                                                       decay_steps,
                                                       decay_rate, staircase=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                               name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy),
                                           tf.contrib.summary.scalar(\"learning rate\", learning_rate)]
            # with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            #     for dataset in [\"dev\", \"test\"]:
            #         self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
            #                                    tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name=\"given_loss\")
                self.given_accuracy = tf.placeholder(tf.float32, [], name=\"given_accuracy\")
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.given_accuracy)]


            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, voxels, labels):
        self.session.run([self.training, self.summaries[\"train\"]], {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy],
                {self.voxels: batch_images, self.labels: batch_labels, self.is_training: False})
            loss += batch_loss * len(batch_images) / len(dataset.voxels)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.voxels)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy

        # accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.voxels: voxels, self.labels: labels, self.is_training: False})
        # return accuracy

    def predict(self, voxels):
        return self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False})


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # test --cnn arg
    # - CB-filters-kernel_size-stride-padding
    # - C-filters-kernel_size-stride-padding
    # - M-kernel_size-stride:
    # - F: Flatten inputs
    # - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
    cnn = \"CB-20-3-1-same,CB-20-3-1-same,CB-10-3-1-same,M-3-2,CB-20-3-1-same,CB-20-3-1-same,CB-10-3-1-same,M-3-2,F,RD-200\"
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=50, type=int, help=\"Batch size.\")
    parser.add_argument(\"--cnn\", default=cnn, type=str, help=\"Description of the CNN architecture.\")
    parser.add_argument(\"--learning_rate_start\", default=0.001, type=float, help=\"Learning rate.\")
    parser.add_argument(\"--learning_rate_finish\", default=0.0001, type=float, help=\"Learning rate decreasing.\")
    parser.add_argument(\"--learning_rate_steps\", default=4, type=int, help=\"How much steps to final learning rate\")
    parser.add_argument(\"--epochs\", default=20, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=8, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--train_split\", default=0.8, type=float, help=\"Ratio of examples to use as train.\")
    parser.add_argument(\"--modelnet_dim\", default=20, type=int, help=\"Dimension of ModelNet data.\")
    args = parser.parse_args()

    if args.learning_rate_steps == -1:
        args.learning_rate_steps = args.epochs

    # Create logdir name
    # args.logdir = \"logs/{}-{}-{}\".format(
    #     os.path.basename(__file__),
    #     datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
    #     \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    # )
    date = datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S_%f\")
    args.logdir = \"logs/{}-{}\".format(
        os.path.basename(__file__),
        date
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset(\"modelnet{}-train.npz\".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset(\"modelnet{}-test.npz\".format(args.modelnet_dim), shuffle_batches=False)

    args.num_examples = train.labels.shape[0]
    print(args.num_examples)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    with open(\"{}/3d_recognition_test_{}_setting.txt\".format(args.logdir, date), \"w\") as setting_file:
        for key, value in sorted(vars(args).items()):
            print(key, value, file=setting_file)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            voxels, labels = train.next_batch(args.batch_size)
            network.train(voxels, labels)

        network.evaluate(\"dev\", dev, args.batch_size)

    # Predict test data
    with open(\"{}/3d_recognition_test.txt\".format(args.logdir), \"w\") as test_file:
        while not test.epoch_finished():
            voxels, _ = test.next_batch(args.batch_size)
            labels = network.predict(voxels)

            for label in labels:
                print(label, file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;0mJxmR$fCgFQ{NIp$+H4A)c@YKj)t3~Y|mM-Kr#hp<wf#seC}S!kxv|BcTL30&*Y&-qQ2^sWw=9a9rUQ-2FLAksP$=j*;vV;+CZrW&Ys+?v4^<ruF~Q?)M=bh&jV&ZV=_AZ8_6aJre5kxHW5^HWS5suM;vmE(s)GqAA(N#6*<`U1x@jBtO9BK}uk1F;Ft7l#oud@!rM>-)A;U=@X9ge_m4uBG_8WBevhhd?Q9L3&SMkGI5P(*C9>Ohu!P3SS(WI4rVcrb6o|?L-G4;A_hMTBH}>X$tssh~Ac?lCyR2w#O&m!VCPRxDErAl@k*qHzSh<v!m$>)0~`kLxUXwVW_b7q+iMrU*3dtnfitZV-?Z5M<`YdePXM#bf99-^GZ}>%KiMeNP9pl27Gf#TY|!go|N=RJPVL_N)H~rN&+C-R7w<k4-#rxNlv<LIK&J(GfyZHer7QOcpZu*9$Hm{W#7U^b)X6V_Eb!ZYlmlB#+oSFHMlMZg9HjMg*>n3?DKMZ4vg+C&sVm=cq?r36(Qk)pSqqE=2Sk)#IIb0Q^~H;{N!P$8FHa?exXQh5Zm|7$~(CBTkAGV@4wHSi6&Z;;7Ezhc9|xbQvS)-Fw+Je`TBoLJ#$i`=&)`fn>?hH!!qc<?x6M`Qjjz82{|z%+%%?*BC6;dl%HH8usP?>DO{DzOLrO~d@O-lN%`yh(fz&hC065XGlC~Nw#f=}_D50l0?d6=I7Dkr$WP45u@paDi&^M4uqm!DKK0V#f%+nJD0Y-ONsAnasI%v@quH?E0>sS1^_*VU)m4($mNyu!HBIPhnhIBCKrKXl`qU|S8wpvi!QroV8S9}YsaM82>>K;M00000{?7&1k405H00FWEq!j=F8U|B@vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
