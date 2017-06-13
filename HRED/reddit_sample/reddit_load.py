import numpy as np
from itertools import chain


def get_reddit_voc(path):
    words = np.load(path + 'wordlist.pkl')
    w_to_i = {v:k for (k,v) in enumerate(words)}
    return words, w_to_i, len(words), np.load(path + 'freqs.pkl')


def load_singles(path, threeD=False):
    train = np.load(path + 'utterances.single.shuffled.train.pkl')
    test = np.load(path + 'utterances.single.shuffled.test.pkl')
    if threeD:
        train = [[d] for d in train]
        test = [[d] for d in train]
    return train, test


def load_pairs(path, threeD=False):
    train = np.load(path + 'utterances.pairs.shuffled.train.pkl')
    test = np.load(path + 'utterances.pairs.shuffled.test.pkl')
    if not threeD:
        train = list(chain(*train))
        test = list(chain(*test))
    return train, test