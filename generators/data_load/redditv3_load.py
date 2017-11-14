import numpy as np
from itertools import chain


def load_pairs_glove6B(path, threeD=True):
    train = np.load(path + 'pairsv3.uniq.censored.glove6B.train.pkl')
    test = np.load(path + 'pairsv3.uniq.censored.glove6B.test.pkl')
    if not threeD:
        train = list(chain(*train))
        test = list(chain(*test))
    return train, test


def load_pairs(path, threeD=True):
    train = np.load(path + 'pairsv3.uniq.censored.train.pkl')
    test = np.load(path + 'pairsv3.uniq.censored.test.pkl')
    if not threeD:
        train = list(chain(*train))
        test = list(chain(*test))
    return train, test


def get_redditv3_voc(path, train_len, pad_value=-1):
    word_list = np.load(path + 'training.dict.pkl')
    freqs = np.array([x[1] for x in word_list] + [train_len])
    total_count = float(freqs.sum())

    idx_to_w = [w[0] for w in word_list]
    w_to_idx = {idx_to_w[i]: i for i in range(len(idx_to_w))}

    w_to_idx[u'<utt_end>'] = pad_value
    idx_to_w.append(u'<utt_end>')

    return idx_to_w, w_to_idx, len(w_to_idx), freqs / total_count
