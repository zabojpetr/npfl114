# coding=utf-8

source_1 = """#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data[\"images\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

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
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name=\"im\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # Create NASNet
            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)
            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                features, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=None, is_training=True)

            # self.classifier_saver = tf.train.import_meta_graph('my_classifier-17000.meta')

            self.nasnet_saver = tf.train.Saver()

            # with tf.name_scope(\"cl\"):
            #     classifier = tf.train.import_meta_graph('my_classifier-17000.meta')
            #
            # # Strip off the \"net2/\" prefix to get the names of the variables in the checkpoint.
            # classifier_saver_varlist = {v.name.lstrip(\"cl/\"): v
            #                 for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope=\"cl/\")}
            # self.classifier_saver = tf.train.Saver(var_list=classifier_saver_varlist)
            #
            # graph = tf.get_default_graph()
            # output_layer = graph.get_tensor_by_name(\"cl/output_layer:0\")


            dense = tf.layers.Dense(2000, tf.nn.relu, name=\"cl/Dense_1\")

            output = tf.layers.Dense(self.LABELS, activation=None, name=\"cl/output_layer\")

            # self.classifier_saver = tf.train.import_meta_graph('my_classifier-340.meta')
            # self.classifier_saver.restore(self.session, tf.train.latest_checkpoint('./'))

            # with tf.name_scope(\"cl\"):
            #     cl = tf.train.import_meta_graph('my_classifier-340.meta')
            #
            # # Strip off the \"net1/\" prefix to get the names of the variables in the checkpoint.
            # cl_varlist = {v.name.lstrip(\"cl/\"): v
            #                 for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope=\"cl/\")}
            # self.classifier_saver = tf.train.Saver(var_list=cl_varlist)
            #
            # graph = tf.get_default_graph()
            # dense = graph.get_tensor_by_name(\"cl/Dense_1:0\")
            # output = graph.get_tensor_by_name(\"cl/output_layer:0\")

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
                                                                           name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.loss),
                                           tf.contrib.summary.scalar(\"train/learning_rate\", learning_rate),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
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

            # Load NASNet
            self.nasnet_saver.restore(self.session, tf.train.latest_checkpoint(\"./finetuning/nasnet3/\"))
            # self.nasnet_saver.restore(self.session, args.nasnet)
            self.classifier_saver = tf.train.Saver(dense.variables + output.variables)
            self.classifier_saver.restore(self.session, tf.train.latest_checkpoint('./finetuning/classifier3/'))
            # self.classifier_saver.restore(self.session, tf.train.latest_checkpoint('./'))





    def train_batch(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]], {self.images: images, self.labels: labels, self.is_training: True})

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


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=25, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=50, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--nasnet\", default=\"nets/nasnet/model.ckpt\", type=str, help=\"NASNet checkpoint path.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--learning_rate_start\", default=0.00001, type=float, help=\"Learning rate.\")
    parser.add_argument(\"--learning_rate_finish\", default=1e-30, type=float, help=\"Learning rate decreasing.\")
    parser.add_argument(\"--learning_rate_steps\", default=25, type=int, help=\"How much steps to final learning rate\")
    args = parser.parse_args()

    if args.learning_rate_steps == -1:
        args.learning_rate_steps = args.epochs

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value)
                  for key, value in sorted(vars(args).items()))).replace(\"/\", \"-\")
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset(\"nsketch-train.npz\")
    dev = Dataset(\"nsketch-dev.npz\", shuffle_batches=False)
    test = Dataset(\"nsketch-test.npz\", shuffle_batches=False)

    args.num_examples = train.images.shape[0]

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels = train.next_batch(args.batch_size)
            network.train_batch(images, labels)

        network.evaluate(\"dev\", dev, args.batch_size)

        network.nasnet_saver.save(network.session, \"./finetuning/nasnet4/nasner\")
        network.classifier_saver.save(network.session, \"./finetuning/classifier4/classifier\")

        # Predict test data
        with open(\"{}/nsketch_transfer_test_{}.txt\".format(args.logdir, str(i)), \"w\") as test_file:
            labels = network.predict(test, args.batch_size)
            for label in labels:
                print(label, file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;3KmOM_m9IjeV^~*rTusWZ9iUyoNuH-S7()qsFl)p_FVt`bIJCnjJq6r?iTy+*$|;ULc3N;DT%nnWkV-fT<MkTlU;x;w!00JZCoq@rrP82fxgynsfHl!RIW)Y;O-e^fIxcbLLJ4PKxx1Yd(6nfmWBXk0WxAy}br0O}MDOGAaXSWtb{R?cKZ*=emc&U*eh2HvyC2x3W%b_McBZP&VU^uVA%g;kl690*a_B9Eu!RvQms;2nkpTb>x6*epq~$wt(;A;z^>u)JWwsd-wvB)#A-gpkBuWNQQ?t>t4tUhzoxyF{f7WH~@g8gT;Y3d9s}>s(&V@%VA0V#|1<|A$^m=;2#GaH`=5@vX5!H3A{LW|EduF&<$}sY0sAG_b#(j1C%B}S7!%MJ~qB*6~DG}kc>G-#2hj_!1(QFqE5I5q`6O&2O=0eo-lPQgJf?A>{u2axao+uf6mW$9Ry@NIXW+Vaw2VhZD=1=&8ObJJCZWuMi2H*Ykf$sBSJu-Eehj~-CGrTE3ijJADV(Xs<WFFSD_Ea@BL9tzu!pI?y0$)c41PPjf5tCct;nA{%DWy4q3jRiQyyvMG#-$(rik&M<!sQ8<lIYP<XzjZ-*sUfk=7hL~bjga*dFnJg5SY%Ik4M;jW23NuG_|iTM84f@vVra<AK`^dT|myaC2iB0QnH0IckiW@e5WeXiOc<8DT(eZ|~3@^5e@O)o|Rm^8p-Vb1LDlc+v1$dMF>z2}3V3bH@9*y)>G5On(`VDTo_CT5}qS_988BW+efcE&J?DkqKSN!r9e@D^T2tvFVTun%N3)&f<yJ?{ROgg8nGN1mM~QuWeSZN_R3%S3UiCXYA>YRje&*!`Q}EUd6y>?fb7lGVCnv4VbD3VAfEF%j59@G-=e>H-fw8d@Mh$MW3xtGNkh8xLdZPKn|H<G8$Hx=lH}0Cl=xoWeF;F*KP5Ub-&t@7R$H@Dp?XX?G!~A#2>nlOH{B$FNm|g6F-a|4k_4N>$}*H(<`sApb9zq$OH4KI4!B_oobBbhlQ`KE|>h7R|ZX-MStEq-c+KCfN-tC*HxfX*{WKBmzPm$y-hojsp2ylZXM<pv%{WO2EH}O}k6Nh5*R2s)@I{9ILH&^Roq%DXO)}0VWz!uYp^zfbg(uJ~eq<nT7Fp?mAzSUUA(5_8dIj5_HKM8!@(&&Khd`g5(MwmSKUJ*~b#6M?_^pu_C!;<1B!KHQ@%B(&$x59xfZ-kDOJP&=-HUWI_r1RVCA*K}pA(OOgv*)=PRcaUwkVfQz~s+IIaICGwrNi$Wb?=l3VLn~u_~Ya$C@_Yc1$p2jWeF<<%}(N_AjxR#~03|oR=76}phngX7s1XRa7XgzmJkoR-?m!Py;s(%GBAVgtjcl3fBaIb(1)x$8N>eQNN)h2)l>ov>+O3TIP^g->wzQw%2Q^LV|1A3MjnAWPESW!jA%1Y@~v?WB7zr;(^-(*R&D~+A|8`dVL(-w09NeNa}lS0xfE0>0NgRCF(ohry)@?a!H;czhlhh+uUaXRhU2CcuaK<FLo(j?&f)$Qg3OpZp~2yi~b<a^MMe=k}BJdQ!OiHo6CCScoDhGgm0@A20MSFoM}PK!e4(RTGYKr7~L*33|*Am@2i=$B?dRr?`ulRqK!gC_N6BYe2+#$}^c87ZM9WlUJImw+$(>@Go}8+(&xzC&zgL(CIG;pZVO;$*+HpwN#`@?7NRp5RTwmb_Dh7rwT+8QK^ZwBzvU6BIxwtG>}WSozv;b0WF>&UgX*l}SX%22y*(NL2}_(WQ8L6Wl>1w>NL*`I$$tpkQgEKKeC8ns`)o;#ag=LzKKCqn51!+S=#pYF=j0^i=qpTLEp-Red-`4d}{-rQ&P`g=;g$ut>~_9p4fNHQbBAj<7p1X+TcU->bbs;v)8A8K+7#r81}E@2F~tWI-^Vk?<(L=h_ewXxG#O`otCGrBcN|NHh*VD!o#@AT-ts)Wy~8#ep6^XMw7XI7q1+;DB{l?<bXsSgZ)Gj3=4&vX(5Je5j-NiwnJaBAI4IB97oy?+}0U^18hio#M{=twsMyV`5wh2h@U#Iy32>OQGl+{U!I@;Gurbj41(%E8Cc{h$d}s4kF4Y-T#)-Fng_5*p{2EEE|K=ipT^J68d)2cm|6uq5$}dBVNGXP-45=$;;YiYf<`MF1D#u2+eail?#0As%;i=0d>ZyvbZ;o_`FR49lviBYhL0?Et>h0Wro7C*`rB)81Xtf*txlW=QDN!FXsG$)+LIpxYHp_Z@;UmZy7H=HCYNs8|;ueifZs@`$R85I<BH^%d%@*T766|*u#^Lm(nDpUx}=eQCqt>IT+qq-$4DKjP94?qmy>C$EgjvpSXH{<2)ckuD3C1Qj?ORPg7n_6RLL)a;D%t^(+^BkoUJw4YX`UaATZv6~!DUuG5ST(Qd#Z96n~oM5UFcQ}ZCM@J~bLWI?3b$3tOX>2vty@uq1O=C|+~iQ>mdcwG!dAYVGOM{ac)`uY#3xUq?Eic?5fB5)&^<s1m8znwE#CRpJ-IRgbIku0yiUG{{gcn!A0p>=#Kf5rjT({UTGR|`_Yrg5tD{zF)A#;fQs0ARoPc0?q-tuaBoyAY(9j~O>L=D&tta)zkE5$&(`1&~>yOiC{Zv<TcPxjRhzm`KR9g=edr<ys~?U_J;&KkpEF;-ygWuKtQ4AH|JpnQlI~FIj4&KEyNXPf3Jjs>Nl&;Nv;?juKCZL6zBLa=|v!C6YUxCa+B0IpyI-@V%$e3iRh350<u$+txw{SSwMZLrwmLUM`khu*5X$HNf?isE?Ish9Emtu;{yEEUY2dilh@`zaSO&mYMWMKn-2NHTs)mbf-6IplekGalE%AhyUPzjpUg%zGYKHzPZ>n!0R{F0#C`4KjN3ZJ2v^uC1w`-9m}xcVp&HPnp~Q+bntt757RJ~d6ZfT({_U0OTc+g>ns#&r!%Q<?VCou?48}4A>Lrk|1PRtgpcQI)OY-Bp+F!~o!M%mHVaPw1UPLw0gsT7BCksMhdF^W3PetAJ@O!k;=}-=;$^Y_&%+-ei6Yv<simT3F1(##CiFXv{06<N;OXRLM~30LWEv4XUTz}s$R|QPA?2q0KeA3?0wosL8t1DxdDmCEn4(Jp^%Q^}Z0vg()N>)JMrPOF#smYlf8@Cm<q6W{g4u6mu+)J}Ge+;^l`4V-P;7@ki?d%j+0n|^iiAwQe#;B@J>+tru^QvIlKKzZYvb6BmBFaTIdO><iH~1^RnHo8rWD`%I9J4kGjyG@_(*nGe^|)1kaJ(U6VMo{Yiu7OQ3CRBf_uC)$*G!t=TmzIOaab4m(T03!KB*lK3Zk=#dXVbSP2>3tj}ohwh(;!801b5ss?Y^j(|irNh4d%A6RkfJgsP~lMyy?^P52fxbrrhH1LUygsahoI4-hqq7t96-a3=_IA{U;vw2<RdV^1LaP7&1E4tt}Wr>)Bt)K~#|Ha|&257NfP;LOX=-=nPdX8RW;0H<xcnl1>%r!gpemLv_zGZOB#6c^oxk(LTBEP<I{?N+I$_xC6#_ZsvKL3Q|bQmNVy63kzk_Y>g=XzI{F*u&d7H87~Fm%Z|7d+f7%t)XQIVA@yUnUR7#kj7R(A6@9%7L(hC@6TrP?eD~v8wp>Vb3H9|M$H;hxoPfmjDZ@TqvYte%_o%qZzpxPHw`07FfSUdP-AHu!@Cv%P)kn6tUEGP4NH#R=`P!rICWJ00H9`v_}8{Tb<xrvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
