import numpy as np
from itertools import chain
import lasagne as L


def split_utt(utt):
    u1, u2, u3 = [i for i,j in enumerate(utt) if j == 1]
    return [utt[:u2], utt[u2:u3], utt[u3:]]


def load_mt(path, split=False, trim=200):
    tr = np.load(path + 'Training.triples.pkl')
    vl = np.load(path + 'Validation.triples.pkl')
    ts = np.load(path + 'Test.triples.pkl')

    if split:
        tr = chain(*map(split_utt, tr))
        vl = chain(*map(split_utt, vl))
        ts = chain(*map(split_utt, ts))

    if trim is not None:
        tr = [utt for utt in tr if len(utt) < trim]
        vl = [utt for utt in vl if len(utt) < trim]
        ts = [utt for utt in ts if len(utt) < trim]

    return tr, vl, ts


def get_mt_voc(path, train_len, pad_value=-1):
    word_list = np.load(path + 'Training.dict.pkl')
    word_list.sort(key=lambda x: x[1])
    freqs = np.array(map(lambda x: x[2], word_list) + [train_len])
    total_count = float(sum(freqs))

    words = map(lambda x: x[:2], word_list)

    w_to_idx = dict(words)
    w_to_idx['<utt_end>'] = pad_value
    idx_to_w = {v: k for (k, v) in w_to_idx.items()}

    return idx_to_w, w_to_idx, len(w_to_idx), freqs / total_count


def get_w2v_embs(path):
    word2vec_embs, word2vec_embs_mask = np.load(path + 'Word2Vec_WordEmb.pkl')
    word2vec_embs = np.vstack([word2vec_embs, L.init.GlorotUniform()((1, 300))]).astype(np.float32)
    word2vec_embs_mask = np.vstack([word2vec_embs_mask, np.ones((1, 300))])

    return word2vec_embs, word2vec_embs_mask