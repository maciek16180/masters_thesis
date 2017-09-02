import numpy as np
from itertools import chain


def split_utt(utt):
    idx = [i for i,j in enumerate(utt) if j == 1]
    if len(idx) < 2:
        return []
    return [utt[:idx[1]], utt[idx[1]:], []]


def load_subtle(path, split=False, trim=200):
    data = np.load(path + 'Subtle_Dataset.triples.pkl')

    if split:
        data = list(chain(*map(split_utt, data)))

    if trim is not None:
        if not split:
            data = [utt for utt in data if len(utt) <= trim]
        else:
            inds_to_remove = set()
            for k in xrange(len(data)):
                if len(data[k]) > trim:
                    for i in xrange(3):
                        inds_to_remove.add(k - (k % 3) + i)
                            
            data = [data[i] for i in xrange(len(data)) if i not in inds_to_remove]
            
    return data
