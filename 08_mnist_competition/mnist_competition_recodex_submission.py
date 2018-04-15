# coding=utf-8

source_1 = """#!/usr/bin/env python3
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
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # TODO: Construct the network and training operation.
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            layers = [self.images]
            for layer in args.cnn.split(','):
                items = layer.split('-')
                if items[0] == 'CB':
                    layers.append(tf.layers.conv2d(layers[-1], int(items[1]), (int(items[2]), int(items[2])),
                                                   (int(items[3]), int(items[3])), items[4], use_bias=False,
                                                   name=\"Conv2D_\" + str(len(layers))))
                    layers.append(tf.nn.relu(tf.layers.batch_normalization(layers[-1], training=self.is_training,
                                                                           name=\"BatchNorm_\" + str(len(layers)))))

                elif items[0] == 'C':
                    layers.append(tf.layers.conv2d(layers[-1], int(items[1]), (int(items[2]), int(items[2])),
                                                   (int(items[3]), int(items[3])), items[4], activation=tf.nn.relu,
                                                   name=\"Conv2D_\" + str(len(layers))))

                elif items[0] == 'M':
                    layers.append(tf.layers.max_pooling2d(layers[-1], (int(items[1]), int(items[1])),
                                                          (int(items[2]), int(items[2])),
                                                          name=\"MaxPool_\" + str(len(layers))))

                elif items[0] == 'F':
                    layers.append(tf.layers.flatten(layers[-1], name=\"Flatten_\" + str(len(layers))))

                elif items[0] == 'R':
                    layers.append(
                        tf.layers.dense(layers[-1], int(items[1]), tf.nn.relu, name=\"Dense_\" + str(len(layers))))

                elif items[0] == 'RD':
                    layers.append(
                        tf.layers.dense(layers[-1], int(items[1]), tf.nn.relu, name=\"Dense_\" + str(len(layers))))
                    layers.append(tf.layers.dropout(layers[-1], training=self.is_training, name=\"Dropout_\" + str(len(layers))))

            features = layers[-1]

            output_layer = tf.layers.dense(features, self.LABELS, activation=None, name=\"output_layer\")
            self.predictions = tf.argmax(output_layer, axis=1)


            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")
            global_step = tf.train.create_global_step()
            # batches to one epoch * epochs/steps
            decay_steps = args.num_examples // args.batch_size * args.epochs / (args.learning_rate_steps+1)
            decay_rate = np.power(args.learning_rate_finish / args.learning_rate_start, 1 / (args.learning_rate_steps))
            learning_rate = tf.train.exponential_decay(args.learning_rate_start, global_step,
                                                       decay_steps,
                                                       decay_rate, staircase=True)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy),
                                           tf.contrib.summary.scalar(\"learning rate\", learning_rate)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        # TODO
        self.session.run([self.training, self.summaries[\"train\"]], {self.images:images, self.labels:labels, self.is_training:True})

    def evaluate(self, dataset, images, labels):
        # TODO
        accuracy, predictions, _ =  self.session.run([self.accuracy, self.predictions, self.summaries[dataset]], {self.images:images, self.labels:labels, self.is_training:False})
        return accuracy, predictions

    def predict(self, dataset, images):
        # TODO
        predictions, _ =  self.session.run([self.predictions, self.summaries[dataset]], {self.images:images, self.labels:[0]*images.shape[0], self.is_training:False})
        return predictions


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
    # cnn = \"CB-10-3-1-same,CB-10-3-1-same,M-3-2,CB-10-3-1-same,CB-10-3-1-same,M-3-2,F,R-100,R-100\"



    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=50, type=int, help=\"Batch size.\")
    parser.add_argument(\"--cnn\", default=cnn, type=str, help=\"Description of the CNN architecture.\")
    parser.add_argument(\"--learning_rate_start\", default=0.001, type=float, help=\"Learning rate.\")
    parser.add_argument(\"--learning_rate_finish\", default=0.001, type=float, help=\"Learning rate decreasing.\")
    parser.add_argument(\"--learning_rate_steps\", default=-1, type=int, help=\"How much steps to final learning rate\")
    parser.add_argument(\"--epochs\", default=1, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=8, type=int, help=\"Maximum number of threads to use.\")
    args = parser.parse_args()


    if args.learning_rate_steps == -1:
        args.learning_rate_steps = args.epochs



    # ODEVZDANE RESENI:
    # args.batch_size = 50
    # args.cnn = \"CB-50-3-1-same,CB-50-3-1-same,M-3-2,CB-50-3-1-same,CB-50-3-1-same,M-3-2,F,R-200\"
    # args.learning_rate_start = 0.002
    # args.learning_rate_finish = 0.000005
    # args.learning_rate_steps = 8
    # args.epochs = 100





    # Create logdir name
    # args.logdir = \"logs/{}-{}-{}\".format(
    #     os.path.basename(__file__),
    #     datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
    #     \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    # )
    date = datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\")
    args.logdir = \"logs/{}-{}\".format(
        os.path.basename(__file__),
        date
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself


    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(\"mnist-gan\", reshape=False, seed=42,
                                            source_url=\"https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/\")

    args.num_examples = mnist.train.num_examples
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)



    with open(\"mnist_competition_test_{}_setting.txt\".format(date), \"w\") as setting_file:
        for key, value in sorted(vars(args).items()):
            print(key, value, file=setting_file)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate(\"dev\", mnist.validation.images, mnist.validation.labels)

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    test_labels = network.predict(\"test\", mnist.test.images)

    with open(\"mnist_competition_test_{}.txt\".format(date), \"w\") as test_file:
        for label in test_labels:
            print(label, file=test_file)"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;B_w+U0nbf13gV=7Yq_8Jt(?mHZYog4pqV6G^cyK9Szyu=and!)1{nG33dLhx%)xq>_3>kdU~a)(0^-?KQi8Nnlbf)8EbDt5U9STBxaRafP_ak9`i;tKO!3gJ&#<m<pHT5!H=)dDcm!M9ozOA{XDpT8LSQmd$^@Zt8J&z!}&M!?{IYPf^5A%d+1-oXG5Az%%Wm}eQ*PwYp%!SF#T>wvA>I;3vEn`r(+&;A)9n_*_lIqganW7*-Q6NX<+w|M?zZ77>T#3eJB81Q@1=6W@rvT-$CFb-EarqSA>wKSF2%kg5N&Xl(|0GL^Pl<=?5R9lxwK>=9BJI(x4fn7FLp%qm<tR?4f=<YU@v(KZ<?o&c_{4j79n^dzAQZotK(ZYc_~?$ch#59j^qa?Ele!-h!oQgUpkV<#CSp(9ZV^-R_D$S<#(A*7vC{D90^VhUFr>TKJ_zv}rl}4<CDluI(O_cq}Z#7t=_jun%j+y!mD(yH@d4IGeMO&o=Ku@_DJP!yVi$Cs<Cin>TK+09c{?l6YFZ9wtQ*fpXG0FpY&FR@5<Aub;fjDaf9((E4dVghE}=YzMC?ndo8=G%mg$-0nT;o9z2c>!cwuLfw@!L1#<211Z3>vOW&UTY#QSg4Ac$JVQZbk&en2>h8+VO!KnvQMU8?fGx*`JP@yoY&bZ^lj?xf4eWwyQ+<kqN<VJd=Eti;G(`_9Z$kO(zLA4Aq2N=+?oi|u6vihW`P@!=nCCZA<W6)uTDR{2ZFJOj>9b*;N+gTD+BxykdU&@9Z+Tt3NtA{(N$YOD!QhzUMJ7P+U%1f>aj;<XH0bIDyr+F0GS(|7!Ol1k?3~pjX!!FEbo9kzN6x0>oOHbHjVX%wg`@{MltK3W4=H;Ds8f3fG|oe(l8&mbO6X-4>-9d=P-ArVy&hE~^Mo2v(vl6j+jg-wl)uVFQHFNJnRlTyM0zPSJrFyccR0N>!79mZaHf_+&hTf{??_KfmF=V|4y7O%?U{vzkNDGEVk{&tJlUipTrQa5+?=$~mIvEn`K<zRJ%{L`Q5UC~9Da!AJ?sjATbr5U4(6lFcuD(!>t$<rU%Ao`C#XBqFE<HGP}Kntf`u}+PF110s(QN1=c5-7Hp>Dl95*`Cd-Z;6B}S#)R6bW7=2o*E4d~|6nJ^^PZ`~%XZ`#1K$#@w9$=B)V87l0R5f7%BAHEIYJ@fRF1DxhASzA-FDzGOKHy0n+C%KL#`r{fZNtN?N>c)~FgKQ==SfBGy<jcPA{q|e3%518fxj<>BsO(7Jhh(5mCCM(v19B80omfdKKi)M@7rF@>&T7qSFH?Er$z_I^7K{?`Pp}j0uyCP~#m0)b&-NreLc@z*^D>F0O`aHWL?r+R#NR{Mxv`6%iPqeO9<BId5!>bGNVypU&*3_KGC4QmUJ+18Jsd!0DUQEPr+yADs_ykgFV0$9E4=CUx}M3=&<3$|E({afkk5ceew+#i(P=C|kIsge2+6OQgx^}9R1}#Nv#J8BP68ueRQW18v5sKWRwX(V7P6r;;+T(#+D9?+sZf}Lp&+Dhj(IvE;;v#Ea?W^8-<QZ#bj;*gT2d0UBREQJmG92;b}hk(p_*wvm7BHdkd0cbBF|C+j}cxgW)JRt$K{h#ssf%#Sutwy6>iArDLNpj*AM-acwKTz0<ki3`7Z_TbZJt*#43UERIcT;6!WgjtSW>t9w~IjJ$nNLT*JH%!4<EFo5^o{LM>kpG~Ug4HkxD3Vl41pQCyEoAP6ti67z*O?1pa6wY#dKl#DHl@LCgzMI%kiZgww7NS&X-pO#{|wtu-Lh;%YB(+l+<Ia?8Y_Ayw)NbUha5sh_eUN>=t3KJYUAjKK)e&&Y^eOgD(ZP&pn0qbV;q()DoJ8_7{#Z`WT7BOYspY|M#^J2~5+bzO{Y1koojv?~NSnyc=WriG5rgHtV!nggoWr3A!rSTDA+p7}af8^)x#B$HDgW;2aVER^8nsC%$*eU~P@?ufwZGwDYQtnMZ5W&A&%GZwlWPj!LgbPgAK}CubOupWon6RElkdycyF7ZXB=PF$Fj+wV*XK7Q1#W*s0CTcDBO#M``Ka;%_uBgif;YUJ28uABEyFIKa4J4PT=#D7d4_#r<HUn-)bZ{Uub$$#ym-QyQf&S@pl(Oox9y6C4Jmg6EfNH>EXpmuMvDto_eccV3Sg5HnM~R##y3<{jWb383c|N_|JN?a8ufDr9wa5yV*i1l_@9f(H(qdFHkxI9Vp(cD?I52K6#ib`+kG>4EOJ_kLElNC8$Zo=H@+wPVr#6IcXU2GZ2<Al%CsxkBExru8EmAzhU20RX_)E6nx-5j#`P3!~dy3>=6e&rc;{1iIIXCU<pBhg4!;<l*VtqzX;ve)J-SB?xKca|@E+tLwDkH2k%{TLw9{*6{{h8%bVOm}Rpw5_DgXO3AY{7LxR=G*`Hu|(r+R~foW}O^Ee;U%}@^ow{%r#qQD(AyqZGN0iT^Jeso6rzta0WUy2!R2aj>8?q77!*9qAwegpL{_GwJXhq;vnxh;ryb$ugr2b#9sTpm_jp@W9H5zm=SQDN1S@P6I3NzX#7Nj=V-}H7=puuzTyyzcec@xCFB^%r<$@bha7Mnsk&~VRs!|Sd<Ldxz_zBB#PsDFWo<(%nT-E`n~j7{6OJ(_SGreeAPYh+Y0zQ8HJ-2Ng{#&^)$eksaqB2;)NNx|dIFyg7qKv#SC!McGH5n(43Cyfr!*MaTEi2`GEcM_ff@Jfs)q=IKG6Ew_58VXYUC^_C-;ll^NNkGs<0xUto52(6!*;9ud-K05JHvnorBe9eNLOVAV}?pvr~E8m{QNz1??z)6%@_}fR7u3Tw7g3+1fFJ_!qWDr`Wp#JG4|J&|<*EqOi-Pl-rEIw0@~1j6!qjS%)OCB^YO2Zq2j9yw>N#Kg}*YaE^p+1>x=Y56y*E?1`@j%v${-t=RAYA*Hq4F@N__CY$Frqf)%*d*faiUu!&}*I>%kgVk31nE7g#e?WG+AqLm;Gs|)Ae&7TvW;k3$%&jcB<d=8Z82v~6+bgH%30Ic(>p=vjxQGJ|N4tW9BE8a3>5YR%Rwn7h_s%WbNuCq$Y>|Va`usc5_t9kyVF72K5O|1d=E{22Y+XqmJ@sYR+_$0J9qg{IT;gK%odAsgk;gmX0OaLEvD1Y{rJ<w1YX-ZftOK{%XNDvI3;UDVo4IT|S=slwa)O~voeGZ<`%S2np%teIr?=zX2C-*o&Z>PWhn^?y;npF5Zwk7&?mDnZDS?KUy9c<00V0Sv1$SwP2+i?{pQt?%c}(||GnYpIGsa$bg`UcgRCiK?#+ob_+EQ{}c*p&z&%B#oSKvsfY2_d(1AJGb%{dR!WjRGramn(t2Fq*pD-29;o3)~^y8OLthQ4yPsFx=vL{I|>)J+l+d{!R~!T;`pw&9%~D#2A3(uzKWlA=`3?p}A&@U)GbI`J}JQ7&EvDXSUPG5vkd(={5r*r|ZpU9CZZrM6d&F_?<&T?(r76*`I?fsuJ|Y;O}uW9FCKTub!_cM6TY(rBRSg~((DTeQ9~*T(b;m;FxCi(d8nH6`-!l^+0qtSblJ<P1{f_a3!nU8IH7X+2n5ee6~m$??+s*71(e0(B3;;WQ~;-NerYz6^HE_JU*xf%Am9w$a~Q0dye!P+|{Iz8K2o89wi#1tv<tLQvb(2lSMz-kFNzpplRRnv998j5!N}eQpry2Ss@U@6Aq$zWJ_1pfe)KJL642aN;?n6BL*sai3p@Az&ona45~~@M_aYbqI0SR=ox6B*%b&Nl`}JK^)d0_KxNP^=is0#7W$gg*Y2bb(vgeA%7IQ(toy=oE^jp^$LQVVN9n2%J)D*8o$2|f&p`9kJC_oN;ajgTQt3s^|5$YVxS^8m56l;iJ5mefxtZvzdK;EYRL!|V{=?-($;<23x32><?RHrlykk)L1*N_)SE@%aP{Q2uA0ho6e0Rt9a9>!RoIjyNtO%h{f1mQwIJ!neKD)9fG7?HY7XFms@=^x+&z`vU&$4z>UV4XEgfghqZRQ1bS%Ldf`6DZS49mkoP1A$V9%9P%e5Kq32EI}txm!3KAdhYQPuN5!Jz}5%e~uOr2s@V<Aj+MPTb`h%wpr=4j}d4z%8A&-Xw<E+OraKi#IU~WZ}=w)#quVx=bgKE@x!N-s}aDF|LrOs!_7cf1PSPZ*PvOfh?9jwU`Z~b>d5P{9cI9g({Q+zhB0YqOD@cDPcieyE;6fOf1t5*&$=6k+F)+Z2oZdiTr^8`5M&q_zS2*yx#ye2q+T|@G@7eRc_%f>{+=C*(xHhB`M*wbP~tYxVpRZ-@>LO8^bpHI~&{6Uj*7k@4a&axlLeG--pDs81awm`2G~<PbL$b)I<AdBE4-i)zLJSA*IruV*q68PNRnJww+3!fpyAQ4p3NV>_S7*q-!Jj<i|DOh7n<&#!o@cYB&&kWpB0;1}f;q`QD&pT9?zI$31Wq_oTD<JUb`H`%s&P_6cE~ALQ3s^B+`^b7!x@zTx<6&Je&9?HsTPR-L5#Qjf5o<C<Q^`7{?f*IPm$BW#ijPrY6p!lVQU-_TnL42+o`Mvo1>ZiEQDNAhMKZ7xgNwyUQldz{v^tUX{`JGwdt$=&UpQ@U4n_@J_?q|KDki`Yg};gt743a2V?s`9!bOt0kMc<J8wPTlP>5K-y6A0vPKpf4}Tn}`M2e@Ifvk0fQr0)GNStylB@5dXH#k?Xs=^OF=@<;8eR4HrIEe<M6moASB6H`s}{?;U5kK!b${ueLUmvzlTE!HTwavN76Nx!+23KhW#APQit_fIBB25wV|^!VjXwVf%dHo#1Oe!{FaNtkW1?7@^O)rQHBw^|M1GWu|RbLgUAWt0F!X8@f8rzSHZ5)xo~YYb$sPP&RR7&RIy|dmVM11R-6u_IZP}ZHi|sx9ON8TaGn{%CfQTB746`A$!_yZrQ77pl-hlEj0RW^JsIGEXjg=W2Lv8iJ=dacUv?StU?}Pyt#3Wn$tu1=`aFy2Dt@ET&7?{uTx*b$t#L@SCXafaR*xbbM(C@??_uCF@0Qs?mE`gtjhO(m|SmBiLAywOB#eelq={fevNgB6RlgtGv5BnE-3$%(YpqlPS?4MHFf6e_>%L7kqOv5?VKuYRohAGewaUs%-dSZhHQeNU0Ib@!=VbFY@TC|K+LL@$kH<?+ZD2~dg593;lpl#49awWUXPh9q(2f^w@w*-9Wx-RAP2|3|5i3*KTt_u4J4fMprUI9@}93)hSUBj1gD6FzN~fNW)S+=l;F^X6;>G(Tol7obuQohi|U?3UE8}H627{qy-Z>tL>YVG9gJT*W$2Prt}k3n;ELeTLOcV^WzVY3M%~MV;&qhK22@G5poFa#R?U^d>R;hDlXOi7Ij)qj=4YVQ(f0gCAnxBj7JtsUO0Z|{TY`3|{Csv68HZ3f{>G$_k(A5UL6_ZrJly5!d!4I3Fa<V59I;C4LC+q>-Y6=T=o@hwqXI;c7cXaf!qqT>;G=<+H}O>wr(22EPG8@h+7Gm`)%rd_K8B?n;6C-3BoN&e6|kMjv*rBqQ2QWVJ@JwxcqG$MdvQPwE;FhP-~7tIevOJ5cc#-YS{nqqhj2L}_`QR=0ycsaop6(3v3SEA*#BpY;jnzFWBq4$&1f}V8J$D=KcAZ&e-q6_=DOlgG^vFP8ZT*<jxDPe#@K}bbCu&@K4y|I7y{Y}kO6n81j3*pw6#owd*!-rv5A!d(m+U{24k%l9?NIy2T{xET0oxs){aF1xvNEHi~aGdl4g;&EhxRh)p%4=zo&svekU@4UqK*q+TU-;k2ai@MS<WS4(lwLKg#+(Ilz9ap}BCOt5HHR0$ozNDTt48Ce)Utx4cVB8j|YHI(ZvYy7MGRND5|!S=hi;4gW~v0uW!Yoi^ahyZK2f;Vk$VVQ)_OymA19rB57{z;@;rMnJu~<68>>*t4!(SJX2Sf%BTZ-G(LRImE2e@s32BSlnRJ+SW>hcR)jnG1>EKft!rEnwk1~t$kbm{EsYVzSLk~;sUNhc^%M0n||q_+K4GU+RMUz2SrHRZRtzr?2NN98qoUcBh3C@@Cicv(CBAqKb|o9(&n;GT+ibuaMAsRU_qywxIx3>h`uHoWFJdP^K%-B@jLFS@=d6C+Ms%v94RzDLS$|8)6N4_-BU=^(uQfXVb3*&ysfK<^gSOVP*#bgAh9h6oEgVz#tyivEAgRX7$hCS%RZHqx-G6OyMEUR6beO?_~a!^WbTU-PuF*1{8!9u!GKPa#@)>F6N#T8Iz*58(ro~A><cp%ck+d2rn!>J*1>5(ly-lMMnWR{dmr_SHKvln(^ev8{`xju`}W)e#0J$Yy&HE<hTH&6K(+EFj=ojle~478Q)0We_yIC?1$^tyV6Gi@rdM5PrlhDg7Lnr(WLQd1mPpAwDOJtFKg#l7L*B=K;`Cj&XuW8fY_N8Y!8lI#wI$dxi9EQb^{YMBJk^n5M_k?{4N3<qbDbwj<-6)cYJ9jGy{j3tJ@;g7oNgFnIF_gazTt?Y{39v81oF+s!uz@P!9rTxztM|LZe#DPe+Gq0=5(j0>}uJ>7~|{G;b5_6B$UE_Q4s7|J+7Y4GWDu_c~DNCBb+S#Qn|*!W~zWO=wt*d7p5xTiphiI-a##y=yoKw+80NVHTeoqh%o1YVT`0<5HKIB)(RdXK-6=jW*`0iY5y7$TVv7lbyyJF>($GI&q{0_30b!2nSYQ(a0|OqW~jpCz_I*cEEst=$z+k@UwgpU&n;>s*>+em<F1M*s)mM#Ve|}HC*)?&p5`c=BbH2*IaXWtEnml#HJz;)CmaG=7&ZCXL4Pe3so)Gb9z;-iiw@y6=(XWGnW)7HH}la3w9g~T$UYLNZAl&gMu!r$!LusI^+=|hHo!rli)V0d^ujre2oyZuVT<0D?M8Yrn8~%;*atTh!zl|azDN->AX{7lz*4#rH9~Y(B_mk7zK22;L#wri9vgbRV=u@wwbMU}NeUAvJ>c*dZcD8~9rm=1^Z9doIFZGWl4CoH?+OkN3?nj*&i+L)2B5+`DJww3Pzk4qBV<5mx%^z-{HrNEPN>@+>O8AEU%;5d#SoL#Q|pmN<R(}m3IUc6Ce<5{=f&i#lLh1r@KR%o2wic_)JV4QeQLh8&8HHU$cvST+q?UUgj!-wTwY7gD8epP$myHwTVR`mXG2X?pAyaR>n$JK*6gvW^z?aGvU{dA^y>DF6w2m|GU2L<-Hyl?!LxqwR-c-e)I~n>*R)?VN4qu2@`tQ_FEQDH_xJYkzVa4*z_a@IJ2WejIkguZCREtYb$aVbNlPcG39Xt<be`v~7aD4_yp`X@bckTN?A}^FUE?o%p{^JJ00000VGeGYtvMn^00H?du<8K-q;%y~vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
