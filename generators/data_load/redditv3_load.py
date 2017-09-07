import numpy as np
from itertools import chain


def load_pairs(path, threeD=True):
    train = np.load(path + 'pairsv3.uniq.censored.train.pkl')
    test = np.load(path + 'pairsv3.uniq.censored.test.pkl')
    if not threeD:
        train = list(chain(*train))
        test = list(chain(*test))
    return train, test