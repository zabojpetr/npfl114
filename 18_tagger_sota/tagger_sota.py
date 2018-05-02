#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import morpho_dataset

class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for i in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[i], columns[i + 1]))
                self.analyses[columns[0]] = analyses

    def get(self, word):
        return self.analyses.get(word, [])


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        # graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, num_tags):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")

            # TODO: Training.
            # Define:
            # - loss in `loss`
            # - training in `self.training`
            # - predictions in `self.predictions`
            # - weights in `weights`

            # TODO(we): Choose RNN cell class according to args.rnn_cell (LSTM and GRU
            # should be supported, using tf.nn.rnn_cell.{BasicLSTM,GRU}Cell).
            if args.rnn_cell == "LSTM":
                fwd = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim, name="LSTM_fwd")
                bwd = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim, name="LSTM_bwd")
            else:
                fwd = tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim, name="GRU_fwd")
                bwd = tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim, name="GRU_bwd")

            # TODO(we): Create word embeddings for num_words of dimensionality args.we_dim
            # using `tf.get_variable`.
            w_embeddings = tf.get_variable("w_embeddings", [num_words, args.we_dim])

            # TODO(we): Embed self.word_ids according to the word embeddings, by utilizing
            # `tf.nn.embedding_lookup`.
            w_embed = tf.nn.embedding_lookup(w_embeddings, self.word_ids)

            # Convolutional word embeddings (CNNE)

            # TODO: Generate character embeddings for num_chars of dimensionality args.cle_dim.
            ch_embedings = tf.get_variable("ch_embedings", [num_chars, args.cle_dim])

            # TODO: Embed self.charseqs (list of unique words in the batch) using the character embeddings.
            ch_embed = tf.nn.embedding_lookup(ch_embedings, self.charseqs)

            # TODO: For kernel sizes of {2..args.cnne_max}, do the following:
            # - use `tf.layers.conv1d` on input embedded characters, with given kernel size
            #   and `args.cnne_filters`; use `VALID` padding, stride 1 and no activation.
            # - perform channel-wise max-pooling over the whole word, generating output
            #   of size `args.cnne_filters` for every word.
            cnns = []
            for kernel in range(2, args.cnne_max + 1):
                layer = tf.layers.conv1d(ch_embed, args.cnne_filters, kernel, name="cnn_" + str(kernel))
                # cnns.append(tf.layers.max_pooling1d(layer, ch_embed.shape[0], 1))
                cnns.append(tf.reduce_max(layer, 1))

            # TODO: Concatenate the computed features (in the order of kernel sizes 2..args.cnne_max).
            # Consequently, each word from `self.charseqs` is represented using convolutional embedding
            # (CNNE) of size `(args.cnne_max-1)*args.cnne_filters`.
            ch_concat = tf.concat(cnns, 1)

            # TODO: Generate CNNEs of all words in the batch by indexing the just computed embeddings
            # by self.charseq_ids (using tf.nn.embedding_lookup).
            cnnes = tf.nn.embedding_lookup(ch_concat, self.charseq_ids)

            # TODO: Concatenate the word embeddings (computed above) and the CNNE (in this order).
            concat_embedings = tf.concat([w_embed, cnnes], 2)

            # TODO(we): Using tf.nn.bidirectional_dynamic_rnn, process the embedded inputs.
            # Use given rnn_cell (different for fwd and bwd direction) and self.sentence_lens.
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fwd, bwd, concat_embedings, self.sentence_lens,
                                                         dtype=tf.float32)

            # TODO(we): Concatenate the outputs for fwd and bwd directions (in the third dimension).
            concat = tf.concat(outputs, 2)

            # TODO(we): Add a dense layer (without activation) into num_tags classes and
            # store result in `output_layer`.
            output_layer = tf.layers.dense(concat, num_tags, name="Dense")

            # TODO(we): Generate `self.predictions`.
            self.predictions = tf.argmax(output_layer, axis=2)

            # TODO(we): Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Training

            # TODO(we): Define `loss` using `tf.losses.sparse_softmax_cross_entropy`, but additionally
            # use `weights` parameter to mask-out invalid words.
            loss = tf.losses.sparse_softmax_cross_entropy(self.tags, output_layer, weights)

            global_step = tf.train.create_global_step()
            decay_steps = args.num_examples // args.batch_size * args.epochs / (args.learning_rate_steps + 1)
            decay_rate = np.power(args.learning_rate_finish / args.learning_rate_start, 1 / (args.learning_rate_steps))
            learning_rate = tf.train.exponential_decay(args.learning_rate_start, global_step,
                                                       decay_steps,
                                                       decay_rate, staircase=True)
            self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/learning_rate", learning_rate),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size, including_charseqs=True)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        tags = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            _,_,pred = self.session.run([self.update_accuracy, self.update_loss, self.predictions],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})
            tags.extend(pred)
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0], tags

    def predict(self, dataset, batch_size):
        tags = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            tags.extend(self.session.run(self.predictions,
                                         {self.sentence_lens: sentence_lens,
                                          self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                                          self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS]}))
        return tags




def analyze(word, tag, dictionary, guesser):
    options = dictionary.get(word) + guesser.get(word)
    if not options:
        return tag
    options = [x.tag for x in options]
    goodness = [sum(1 for i in range(min(len(x),len(tag))) if x[i] == tag[i]) for x in options]
    am = np.argmax(goodness)
    if goodness[am] >= len(options[am])*0: #shoda vice nez 50%
        return options[am]
    else:
        return tag


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    # parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    # parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=32, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--cnne_filters", default=16, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnne_max", default=4, type=int, help="Maximum CNN filter length.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
    parser.add_argument("--learning_rate_start", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_finish", default=0.0001, type=float, help="Learning rate decreasing.")
    parser.add_argument("--learning_rate_steps", default=-1, type=int, help="How much steps to final learning rate")
    args = parser.parse_args()

    if args.learning_rate_steps == -1:
        args.learning_rate_steps = args.epochs

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
    dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", train=train, shuffle_batches=False)

    analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")

    args.num_examples = len(train.sentence_lens)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      len(train.factors[train.TAGS].words))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)
        acc, pred = network.evaluate("dev", dev, args.batch_size)
        print(acc)

        with open("{}/tagger_sota_dev_{}.txt".format(args.logdir, i), "w", encoding="utf-8") as dev_file:
            forms = dev.factors[dev.FORMS].strings
            tags = pred
            for s in range(len(forms)):
                for j in range(len(forms[s])):



                    print("{}\t_\t{}".format(forms[s][j], dev.factors[dev.TAGS].words[tags[s][j]]) , file=dev_file)
                print("", file=dev_file)

        # Predict test data
        with open("{}/tagger_sota_test_{}.txt".format(args.logdir, i), "w", encoding="utf-8") as test_file:
            forms = test.factors[test.FORMS].strings
            tags = network.predict(test, args.batch_size)
            for s in range(len(forms)):
                for j in range(len(forms[s])):
                    print("{}\t_\t{}".format(forms[s][j], test.factors[test.TAGS].words[tags[s][j]]), file=test_file)
                print("", file=test_file)
        # Predict test data
        with open("{}/tagger_sota_test_analyzer_{}.txt".format(args.logdir, i), "w", encoding="utf-8") as test_file:
            forms = test.factors[test.FORMS].strings
            tags = network.predict(test, args.batch_size)
            for s in range(len(forms)):
                for j in range(len(forms[s])):
                    print("{}\t_\t{}".format(forms[s][j], analyze(forms[s][j], test.factors[test.TAGS].words[tags[s][j]], analyzer_dictionary, analyzer_guesser)), file=test_file)
                print("", file=test_file)
