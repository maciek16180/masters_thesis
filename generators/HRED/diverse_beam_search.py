# for use with HRED

import numpy as np
import sys


def softmax(x):
    assert len(x.shape) == 2
    x = np.exp(x)
    return x / x.sum(axis=1)[:, np.newaxis]


class DiverseBeamSearch(object):

    def __init__(self, words, model, beam_size, group_size,
                 rank_penalty=0,
                 group_diversity_penalty=1,
                 seq_diversity_penalty=1,
                 unk_penalty=100,
                 sharpen_probs=None,
                 random_sample=False,
                 only_last_groups=False,
                 verbose_log=False,
                 forbidden_words=['<unk>','<number>','<person>','<continued_utterance>']):

        assert not beam_size % group_size

        self.model = model
        self.idx_to_w = words
        self.voc_size = len(words)
        self.w_to_idx = {words[i] : i for i in xrange(self.voc_size)}
        self.forbidden_words = [w for w in forbidden_words if w in self.w_to_idx]

        self.beam_size = beam_size
        self.group_size = group_size

        self.rank_penalty = rank_penalty
        self.group_diversity_penalty = group_diversity_penalty
        self.seq_diversity_penalty = seq_diversity_penalty
        self.unk_penalty = unk_penalty

        self.verbose_log = verbose_log
        self.random_sample = random_sample
        self.only_last_groups = only_last_groups

        self.sharpen_probs = sharpen_probs


    def search(self, dec_init, init_seq=None):

        if init_seq is None:
            init_seq = np.array([[self.w_to_idx['<s>']]])
        num_groups = self.beam_size / self.group_size

        seq = np.repeat(init_seq.astype(np.int32), self.beam_size, axis=0)

        scores = np.zeros(self.beam_size)

        finished = []

        while seq.shape[1] < 50:
            all_probs, all_dec_init = self.model.get_probs_and_new_dec_init_fn(seq[:,-1:], dec_init)

            new_seq = np.zeros((0, seq.shape[1] + 1), dtype=np.int32)
            new_dec_inits = []
            next_scores = []

            for g in xrange(num_groups):
                g_idx = slice(self.group_size * g, self.group_size * (g + 1))
                log_probs = np.log(all_probs[g_idx])

                if self.unk_penalty is not None:
                    for w in self.forbidden_words:
                        log_probs[:, self.w_to_idx[w]] -= self.unk_penalty

                dec_init = all_dec_init[g_idx]

                # here we add the dissimilarity as described in https://arxiv.org/pdf/1610.02424.pdf
                # simple Hamming diversity
                log_probs[:, new_seq[:, -1]] -= self.group_diversity_penalty

                # penalize repeating words in the same sequence
                log_probs[np.indices((self.group_size, seq.shape[1]))[0], seq[g_idx]] -= self.seq_diversity_penalty

                words = np.arange(self.voc_size)[np.newaxis].repeat(self.group_size, axis=0).astype(np.int32)

                next_word_scores = log_probs[np.indices((self.group_size, self.voc_size))[0], words]

                new_scores = next_word_scores + scores[g_idx, np.newaxis]

                # this line is for implementing rank penalty: https://arxiv.org/abs/1611.08562
                # it doesn't work correctly at the moment
                new_scores = (new_scores - (new_scores.argsort(axis=1) + 1) * self.rank_penalty).ravel()

                new_scores = new_scores.ravel()

                if not self.random_sample:
                    if seq.shape[1] == 1:
                        order = np.array(-new_scores[:self.voc_size]).argsort().astype(np.int32)
                    else:
                        order = (-new_scores).argsort().astype(np.int32)
                else:
                    count = new_scores.size
                    scr = new_scores if self.sharpen_probs is None else -(-new_scores**self.sharpen_probs)
                    order = np.random.choice(count, size=2*self.group_size**2, replace=False, p=softmax(scr[np.newaxis])[0])

                #print "###############"
                #print new_seq.shape, order.size, new_scores.shape, next_word_scores.shape, scores.shape

                for idx in order:
                    if new_seq.shape[0] == (g + 1) * self.group_size:
                        break

                    i,j = divmod(idx, self.voc_size)

                    #print seq.shape, self.group_size * g + i, i, j
                    #print words

                    extended_seq = np.concatenate([seq[self.group_size * g + i], np.array([words[i,j]])])
                    if extended_seq[-1] == self.w_to_idx['</s>']:
                        if not self.only_last_groups or g >= num_groups / 2:
                            finished.append((extended_seq, new_scores[idx]))
                    else:
                        new_seq = np.vstack([new_seq, extended_seq])
                        new_dec_inits.append(dec_init[i])
                        next_scores.append(new_scores[idx])

                if new_seq.shape[0] < (g + 1) * self.group_size:
                    pad_len = (g + 1) * self.group_size - new_seq.shape[0]
                    new_seq = np.vstack([new_seq, np.ones((pad_len, new_seq.shape[1]))]).astype(np.int32)
                    new_dec_inits.append(np.zeros((pad_len, dec_init.shape[1]), dtype=np.float32))
                    next_scores += [-np.inf] * pad_len

            if not new_seq.size:
                print 'Ending...'
                break

            seq = new_seq
            scores = np.array(next_scores)
            dec_init = np.vstack(new_dec_inits)

            if self.verbose_log:
                print 'Length ', seq.shape[1], '\n'
                for utt, s in zip(seq, scores):
                    print '{:.4f} {}'.format(s, print_utt(utt))
                    print ''
                print '#############\n'

    #     final_scores = np.array(map(lambda x: x[1], finished))
    #     finished = map(lambda x: x[0], finished)

        return finished#[final_scores.argmax()]
