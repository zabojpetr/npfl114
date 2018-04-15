#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._permutation = np.random.permutation(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm], self._masks[batch_perm]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images))
            return True
        return False

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
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.my_masks = tf.placeholder(tf.int64, [None, self.HEIGHT, self.WIDTH, self.LABELS])
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.labels_predictions` of shape [None] and type tf.int64
            # - mask predictions are stored in `self.masks_predictions` of shape [None, 28, 28, 1] and type tf.float32
            #   with values 0 or 1

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
                    layers.append(
                        tf.layers.dropout(layers[-1], training=self.is_training, name="Dropout_" + str(len(layers))))


            output_layer = tf.layers.dense(layers[-1], self.LABELS * 28*28*2, activation=None, name='output_layer')
            # rs = tf.map_fn(lambda x: tf.argmax(x, axis=-1) , tf.reshape(output_layer,(-1,self.HEIGHT, self.WIDTH, self.LABELS,2)))
            rs = tf.argmax(tf.reshape(output_layer,(-1,self.HEIGHT, self.WIDTH, self.LABELS,2)), axis=-1)
            print("rs ", rs.shape)
            rsum = tf.reduce_sum(rs,axis=(1,2))

            self.labels_predictions = tf.argmax(rsum, axis=1)
            self.mp =  tf.argmax(tf.reshape(output_layer,(-1,self.HEIGHT, self.WIDTH, self.LABELS,2)), axis=-1)
            self.masks_predictions = tf.cast(tf.expand_dims(tf.map_fn(lambda x: x[:,:,tf.argmax(tf.reduce_sum(x,axis=(0,1)))], self.mp),-1), tf.float32)
            print(self.masks_predictions.shape)
            print(self.masks_predictions.dtype)

            loss = tf.losses.sparse_softmax_cross_entropy(self.my_masks, tf.reshape(output_layer,(-1,28,28,1,2)))
            global_step = tf.train.create_global_step()# batches to one epoch * epochs/steps
            decay_steps = args.num_examples // args.batch_size * args.epochs / (args.learning_rate_steps+1)
            decay_rate = np.power(args.learning_rate_finish / args.learning_rate_start, 1 / (args.learning_rate_steps))
            learning_rate = tf.train.exponential_decay(args.learning_rate_start, global_step,
                                                       decay_steps,
                                                       decay_rate, staircase=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                               name="training")

            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1,2,3])
            iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", accuracy),
                                           tf.contrib.summary.scalar("train/iou", iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions),
                                           tf.contrib.summary.scalar("learning rate", learning_rate)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset+"/loss", loss),
                                               tf.contrib.summary.scalar(dataset+"/accuracy", accuracy),
                                               tf.contrib.summary.scalar(dataset+"/iou", iou),
                                               tf.contrib.summary.image(dataset+"/images", self.images),
                                               tf.contrib.summary.image(dataset+"/masks", self.masks_predictions)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, masks):
        my_masks = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], self.LABELS))
        for i in range(my_masks.shape[0]):
            my_masks[i, :, :, labels[i]] = masks[i, :, :, 0]
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.my_masks: my_masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        my_masks = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], self.LABELS))
        for i in range(my_masks.shape[0]):
            my_masks[i, :, :, labels[i]] = masks[i, :, :, 0]
        self.session.run(self.summaries[dataset],
                         {self.images: images, self.labels: labels, self.masks: masks, self.my_masks: my_masks, self.is_training: False})

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})


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
    # args.cnn = "CB-50-3-1-same,CB-50-3-1-same,M-3-2,CB-50-3-1-same,CB-50-3-1-same,M-3-2,F,R-7840"
    # args.learning_rate_start = 0.001
    # args.learning_rate_finish = 0.0001
    # args.learning_rate_steps = 50
    # args.epochs = 50



    # Create logdir name
    # args.logdir = "logs/{}-{}-{}".format(
    #     os.path.basename(__file__),
    #     datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    #     ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
    #               for key, value in sorted(vars(args).items()))).replace("/", "-")
    # )
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    args.logdir = "logs/{}-{}".format(
        os.path.basename(__file__),
        date
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz")


    args.num_examples = train.labels.shape[0]

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    with open("sparse_fashion_masks_test_{}_setting.txt".format(date), "w") as setting_file:
        for key, value in sorted(vars(args).items()):
            print(key, value, file=setting_file)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels, masks = train.next_batch(args.batch_size)
            network.train(images, labels, masks)

        # while not dev.epoch_finished():
        #     images, labels, masks = dev.next_batch(args.batch_size)
        #     network.evaluate("dev", images, labels, masks)
        network.evaluate("dev", dev.images, dev.labels, dev.masks)

    labels, masks = network.predict(test.images)
    with open("sparse_fashion_masks_test_{}.txt".format(date), "w") as test_file:
        for i in range(len(labels)):
            print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)
