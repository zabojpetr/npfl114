#!/usr/bin/env python3
import numpy as np

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

def analyze(word, tag, dictionary, guesser):
    options = dictionary.get(word)
    if not options:
        options = guesser.get(word)
    options = [x.tag for x in options]

    if not options:
        return tag
    if len(options) == 1:
        return options[0]

    goodness = [sum(1 for i in range(min(len(x),len(tag))) if x[i] == tag[i]) for x in options]
    am = np.argmax(goodness)

    idx = [i for i,x in enumerate(goodness) if x == goodness[am]]
    try:
        idx.index(options.index(tag))
        return tag
    except:
        o = options[am]
        return o


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    # np.random.seed(42)

    # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    # parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    # parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser = argparse.ArgumentParser()
    parser.add_argument("best", type=str, help="dev.")
    parser.add_argument("prediction",  type=str, help="prediction.")
    args = parser.parse_args()

    analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")

    prediction = morpho_dataset.MorphoDataset(args.prediction)

    dir = os.path.dirname(args.prediction)
    f = os.path.basename(args.prediction)

    with open("{}/a_{}.txt".format(dir, f), "w", encoding="utf-8") as test_file:
        forms = prediction.factors[prediction.FORMS].strings
        tags = prediction.factors[prediction.TAGS].strings
        for s in range(len(forms)):
            for j in range(len(forms[s])):
                print("{}\t_\t{}".format(forms[s][j], analyze(forms[s][j], tags[s][j], analyzer_dictionary, analyzer_guesser)), file=test_file)
            print("", file=test_file)

    os.system('python morpho_eval.py ' + args.best + " " + args.prediction)
    os.system('python morpho_eval.py ' + args.best + " " + "{}/a_{}.txt".format(dir, f))
