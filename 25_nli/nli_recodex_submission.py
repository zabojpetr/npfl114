# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import nli_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, num_languages):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name=\"sentence_lens\")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name=\"word_ids\")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name=\"charseqs\")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name=\"charseq_lens\")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name=\"charseq_ids\")
            self.languages = tf.placeholder(tf.int32, [None], name=\"languages\")

            # TODO: Training.
            # Define:
            # - loss in `loss`
            # - training in `self.training`
            # - predictions in `self.predictions`


            # TODO(we): Choose RNN cell class according to args.rnn_cell (LSTM and GRU
            # should be supported, using tf.nn.rnn_cell.{BasicLSTM,GRU}Cell).
            if args.rnn_cell == \"LSTM\":
                fwd = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim, name=\"LSTM_fwd\")
                bwd = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim, name=\"LSTM_bwd\")
            else:
                fwd = tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim, name=\"GRU_fwd\")
                bwd = tf.nn.rnn_cell.GRUCell(args.rnn_cell_dim, name=\"GRU_bwd\")

            # TODO(we): Create word embeddings for num_words of dimensionality args.we_dim
            # using `tf.get_variable`.
            w_embeddings = tf.get_variable(\"w_embeddings\", [num_words, args.we_dim])

            # TODO(we): Embed self.word_ids according to the word embeddings, by utilizing
            # `tf.nn.embedding_lookup`.
            w_embed = tf.nn.embedding_lookup(w_embeddings, self.word_ids)

            # Convolutional word embeddings (CNNE)

            # TODO: Generate character embeddings for num_chars of dimensionality args.cle_dim.
            ch_embedings = tf.get_variable(\"ch_embedings\", [num_chars, args.cle_dim])

            # TODO: Embed self.charseqs (list of unique words in the batch) using the character embeddings.
            ch_embed = tf.nn.embedding_lookup(ch_embedings, self.charseqs)

            # TODO: For kernel sizes of {2..args.cnne_max}, do the following:
            # - use `tf.layers.conv1d` on input embedded characters, with given kernel size
            #   and `args.cnne_filters`; use `VALID` padding, stride 1 and no activation.
            # - perform channel-wise max-pooling over the whole word, generating output
            #   of size `args.cnne_filters` for every word.
            cnns = []
            for kernel in range(2, args.cnne_max+1):
                layer = tf.layers.conv1d(ch_embed, args.cnne_filters, kernel, name=\"cnn_\"+str(kernel))
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
            output_layer = tf.layers.dense(concat, num_languages, name=\"Dense\")
            print(output_layer.shape)

            # tf.layers.dense(output_layer, )

            # TODO(we): Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            weights = tf.expand_dims(weights, -1)
            weights = tf.concat([weights]*11, 2)
            print(weights.shape)


            # output_layer = tf.reduce_mean(output_layer, 1)
            output_layer = tf.reduce_sum(weights * output_layer, 1)/tf.reduce_sum(weights, 1)


            # TODO(we): Generate `self.predictions`.
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training

            # TODO(we): Define `loss` using `tf.losses.sparse_softmax_cross_entropy`, but additionally
            # use `weights` parameter to mask-out invalid words.
            loss = tf.losses.sparse_softmax_cross_entropy(self.languages, output_layer)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name=\"training\")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.languages, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(loss)
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.update_loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \\
                train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries[\"train\"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.languages: languages})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \\
                dataset.next_batch(batch_size)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.languages: languages})

        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        languages = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, _ = \\
                dataset.next_batch(batch_size)
            languages.extend(self.session.run(self.predictions,
                                              {self.sentence_lens: sentence_lens,
                                               self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                                               self.word_ids: word_ids, self.charseq_ids: charseq_ids}))

        return languages


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument(\"--batch_size\", default=None, type=int, help=\"Batch size.\")
    # parser.add_argument(\"--epochs\", default=None, type=int, help=\"Number of epochs.\")
    # parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
    # args = parser.parse_args()


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=10, type=int, help=\"Batch size.\")
    parser.add_argument(\"--cle_dim\", default=128, type=int, help=\"Character-level embedding dimension.\")
    parser.add_argument(\"--cnne_filters\", default=64, type=int, help=\"CNN embedding filters per length.\")
    parser.add_argument(\"--cnne_max\", default=8, type=int, help=\"Maximum CNN filter length.\")
    parser.add_argument(\"--epochs\", default=50, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--recodex\", default=False, action=\"store_true\", help=\"ReCodEx mode.\")
    parser.add_argument(\"--rnn_cell\", default=\"LSTM\", type=str, help=\"RNN cell type.\")
    parser.add_argument(\"--rnn_cell_dim\", default=256, type=int, help=\"RNN cell dimension.\")
    parser.add_argument(\"--threads\", default=8, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--we_dim\", default=256, type=int, help=\"Word embedding dimension.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    train = nli_dataset.NLIDataset(\"nli-train.txt\")
    dev = nli_dataset.NLIDataset(\"nli-dev.txt\", train=train, shuffle_batches=False)
    test = nli_dataset.NLIDataset(\"nli-test.txt\", train=train, shuffle_batches=False)

    sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \\
        train.next_batch(args.batch_size)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.vocabulary(\"words\")), len(train.vocabulary(\"chars\")), len(train.vocabulary(\"languages\")))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        acc = network.evaluate(\"dev\", dev, args.batch_size)
        print(acc)

    # Predict test data
        with open(\"{}/nli_test_{}.txt\".format(args.logdir, i), \"w\", encoding=\"utf-8\") as test_file:
            languages = network.predict(test, args.batch_size)
            for language in languages:
                print(test.vocabulary(\"languages\")[language], file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;1zoVxLp7x6b0JSbk+(<3hV>VJY^C+{RhPqS7Z=~z8|wf+O7dS-;DlF*s;LaQ~Zzh%JDR2K^Gcdo%5n+6O^r3^T7FPyh;{&@GFuDDaq-X?J+-^<w*(aoPUn!nD5xD`%i|b`2tiIOiM!5J!G!f5;|yPMr&jP+KU}ZC<y>`T!t+%7OGFP>}g}l1zWD2kRH<q*TQ}j2a%th2h3SYQt)V%8djW(oo<}X52x;$>KvBu7TpDFhL-|gE!J$GVS$x4@(6N^BD=v$0?{<RSEQh28%x<|^GX<qM55ulQ7rx5*|y7;6eeF*%|YksL9Lp5$B+;usz#|$4B{l+@$=dK*`a1&gvOL4^hhs;U_^G3RF(_>$RD`LJ`?e>aU$#ov{1gt>g>8#AYQ+5o;ZHy!nc9irN>>%T-x!Y^ea|<`A%DRUll~JLBy|5<7W#$2%L^O^e!mXQBbT^<65``isaUBd}0`t->Wbu3#IfJI&gi<K=?89ZRsnsQft)WiNldfEEj56w2}My5At+~cNLjS?7PJTe@vtS;Jo=9u`pWt4SZ~6Q@$WR-hPotZLSUz1QTQ;@1|SF-qtQ9;@bDUCGa|Lui`bAwc<mmzc~%;M$fE<51@Lx09vyi82-G;GAIJ2BTT<JaQ?^KtoE?vEH0?H=*`Ym0wY2ADFOFxWy5oTOC$gSPL#%cOXSUutbxVn)TXDE?r7V^$(SKA?~QBso;&n2UT7^y8eVhs^u12u{CEa865a<Bs<^7&*B!5AV^>N%Ng-Wwg4*=Tffv3iDuVVTaO!#NJ%tqrlg<L_trxf>^D=X^fO8ok*q|{O{$**S&l{t&ENyksChyL_zJawo(Y64f-*!6)?RpaI4h9L#p!Ea*;IilZlVN<KGJMI^sA0*?>hRsQeVeY<5>D;-Ewiu7(6td;<qZgBNJZSG`Ompqow1Fz@A4RGH$N#!Fsg1j|IQY-x`kgTFg2T9@Ex4kr1F7{98WBJ`i;G1*1=Hu_W^Ncjb4}uyb-aZek3DMirjE|W|1#wqb=_$n)=O@jRTh^7w!^QH7?ghLYs@-@TWc(TChJd!>4U2KEg%oo{1*Ni;nadP&tXqhwz?R4ZOL87566TuW0r&GuN-g5s`0hsxD{I>WPWrX`F-^C;d|JD*{3E8qsXh&w^C5zr^LEDZV<pv7rO~8R6!G*<?U#j&|r;ihbNerMiu+_Zb;6h$4Vm$#tQG`^|}9)&^%wz?%jD+ED1FyF1v_00Gno{3-wd-8`@XvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
