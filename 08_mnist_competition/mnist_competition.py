#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph = graph, config=config)

    def construct(self, args):
        with self.session.graph.as_default():
            # TODO: Construct the network and training operation.
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            layers = [self.images]
            for layer in args.cnn.split(','):
                items = layer.split('-')
                if items[0] == 'CB':
                    layers.append(tf.layers.conv2d(layers[-1], int(items[1]), (int(items[2]), int(items[2])),
                                                   (int(items[3]), int(items[3])), items[4], use_bias=False,
                                                   name="Conv2D_" + str(len(layers))))
                    layers.append(tf.nn.relu(tf.layers.batch_normalization(layers[-1], training=self.is_training,
                                                                           name="BatchNorm_" + str(len(layers)))))

                elif items[0] == 'C':
                    layers.append(tf.layers.conv2d(layers[-1], int(items[1]), (int(items[2]), int(items[2])),
                                                   (int(items[3]), int(items[3])), items[4], activation=tf.nn.relu,
                                                   name="Conv2D_" + str(len(layers))))

                elif items[0] == 'M':
                    layers.append(tf.layers.max_pooling2d(layers[-1], (int(items[1]), int(items[1])),
                                                          (int(items[2]), int(items[2])),
                                                          name="MaxPool_" + str(len(layers))))

                elif items[0] == 'F':
                    layers.append(tf.layers.flatten(layers[-1], name="Flatten_" + str(len(layers))))

                elif items[0] == 'R':
                    layers.append(
                        tf.layers.dense(layers[-1], int(items[1]), tf.nn.relu, name="Dense_" + str(len(layers))))

                elif items[0] == 'RD':
                    layers.append(
                        tf.layers.dense(layers[-1], int(items[1]), tf.nn.relu, name="Dense_" + str(len(layers))))
                    layers.append(tf.layers.dropout(layers[-1], training=self.is_training, name="Dropout_" + str(len(layers))))

            features = layers[-1]

            output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)


            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            # batches to one epoch * epochs/steps
            decay_steps = args.num_examples // args.batch_size * args.epochs / (args.learning_rate_steps+1)
            decay_rate = np.power(args.learning_rate_finish / args.learning_rate_start, 1 / (args.learning_rate_steps))
            learning_rate = tf.train.exponential_decay(args.learning_rate_start, global_step,
                                                       decay_steps,
                                                       decay_rate, staircase=True)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),
                                           tf.contrib.summary.scalar("learning rate", learning_rate)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        # TODO
        self.session.run([self.training, self.summaries["train"]], {self.images:images, self.labels:labels, self.is_training:True})

    def evaluate(self, dataset, images, labels):
        # TODO
        accuracy, predictions, _ =  self.session.run([self.accuracy, self.predictions, self.summaries[dataset]], {self.images:images, self.labels:labels, self.is_training:False})
        return accuracy, predictions

    def predict(self, dataset, images):
        # TODO
        predictions, _ =  self.session.run([self.predictions, self.summaries[dataset]], {self.images:images, self.labels:[0]*images.shape[0], self.is_training:False})
        return predictions


if __name__ == "__main__":
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
    cnn = "CB-10-3-1-same,CB-10-3-1-same,M-3-2,CB-10-3-1-same,CB-10-3-1-same,M-3-2,F,R-100,R-100"



    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default=cnn, type=str, help="Description of the CNN architecture.")
    parser.add_argument("--learning_rate_start", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_finish", default=0.001, type=float, help="Learning rate decreasing.")
    parser.add_argument("--learning_rate_steps", default=-1, type=int, help="How much steps to final learning rate")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()


    if args.learning_rate_steps == -1:
        args.learning_rate_steps = args.epochs



    # ODEVZDANE RESENI:
    # args.batch_size = 50
    # args.cnn = "CB-50-3-1-same,CB-50-3-1-same,M-3-2,CB-50-3-1-same,CB-50-3-1-same,M-3-2,F,R-200"
    # args.learning_rate_start = 0.002
    # args.learning_rate_finish = 0.000005
    # args.learning_rate_steps = 8
    # args.epochs = 100





    # Create logdir name
    # args.logdir = "logs/{}-{}-{}".format(
    #     os.path.basename(__file__),
    #     datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    #     ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    # )
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.logdir = "logs/{}-{}".format(
        os.path.basename(__file__),
        date
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself


    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    args.num_examples = mnist.train.num_examples
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)



    with open("mnist_competition_test_{}_setting.txt".format(date), "w") as setting_file:
        for key, value in sorted(vars(args).items()):
            print(key, value, file=setting_file)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        ve = mnist.validation.epochs_completed
        while mnist.validation.epochs_completed == ve:
            images, labels = mnist.validation.next_batch(100)
            network.evaluate("dev", images, labels)
        #network.evaluate("dev", mnist.validation.images, mnist.validation.labels)

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    test_labels = network.predict("test", mnist.test.images)

    with open("mnist_competition_test_{}.txt".format(date), "w") as test_file:
        for label in test_labels:
            print(label, file=test_file)