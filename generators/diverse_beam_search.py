from __future__ import print_function

import numpy as np
import sys


forbidden_words = [
    '<unk>',
    '<number>',
    '<person>',
    '<continued_utterance>'
]


def softmax(x):
    assert len(x.shape) == 2
    x = np.exp(x)
    return x / x.sum(axis=1)[:, np.newaxis]


def go_down_trie(trie, seq):
    for x in seq:
        if x not in trie:
            raise KeyError("Sequence is not in trie.")
        trie = trie[x]
    return trie


class DiverseBeamSearch(object):

    def __init__(self, words, model, beam_size, group_size,
                 rank_penalty=0,
                 group_diversity_penalty=1,
                 seq_diversity_penalty=1,
                 unk_penalty=100,
                 random_sample=False,
                 verbose_log=False,
                 whitelist=None,
                 forbidden_words=forbidden_words):

        assert not beam_size % group_size

        self.model = model
        self.voc_size = len(words)
        self.w_to_idx = {words[i]: i for i in range(self.voc_size)}
        self.forbidden_words = [
            self.w_to_idx[w] for w in forbidden_words if w in self.w_to_idx]

        self.beam_size = beam_size
        self.group_size = group_size

        self.rank_penalty = rank_penalty
        self.group_diversity_penalty = group_diversity_penalty
        self.seq_diversity_penalty = seq_diversity_penalty
        self.unk_penalty = unk_penalty

        self.verbose_log = verbose_log
        self.random_sample = random_sample

        self.whitelist = whitelist

    def search(self, dec_init, init_seq=None):

        if init_seq is None:
            init_seq = np.array([[self.w_to_idx['<s>']]])
        num_groups = self.beam_size / self.group_size

        seq = np.repeat(init_seq.astype(np.int32), self.beam_size, axis=0)

        scores = np.zeros(self.beam_size)

        finished = []
        finished_dec_inits = []

        if self.whitelist is not None:
            trie_positions = [
                self.whitelist[self.w_to_idx['<s>']]] * self.beam_size

        while seq.shape[1] < 50:
            all_probs, all_dec_init = self.model.get_probs_and_new_dec_init_fn(
                seq[:, -1:], dec_init)

            new_seq = np.zeros((0, seq.shape[1] + 1), dtype=np.int32)
            new_dec_inits = []
            next_scores = []

            for g in range(num_groups):
                g_idx = slice(self.group_size * g, self.group_size * (g + 1))
                log_probs = np.log(all_probs[g_idx])

                if self.unk_penalty is not None:
                    for w in self.forbidden_words:
                        log_probs[:, w] -= self.unk_penalty

                dec_init = all_dec_init[g_idx]

                # Here we add the dissimilarity as described in
                # https://arxiv.org/pdf/1610.02424.pdf
                # simple Hamming diversity
                for used_w in new_seq[:, -1]:
                    log_probs[:, used_w] -= self.group_diversity_penalty

                # penalize repeating words in the same sequence
                log_probs[np.indices((self.group_size, seq.shape[1]))[0],
                          seq[g_idx]] -= self.seq_diversity_penalty

                words = np.arange(self.voc_size)[np.newaxis].repeat(
                    self.group_size, axis=0).astype(np.int32)

                next_word_scores = log_probs[
                    np.indices((self.group_size, self.voc_size))[0], words]

                new_scores = next_word_scores + scores[g_idx, np.newaxis]

                # This line is for implementing rank penalty:
                # https://arxiv.org/abs/1611.08562
                # it doesn't work correctly at the moment
                new_scores = new_scores - (
                    new_scores.argsort(axis=1) + 1) * self.rank_penalty

                if self.whitelist is not None:
                    cands = []
                    for i in range(self.group_size):
                        cands.append(
                            trie_positions[self.group_size * g + i].keys())

                    cand_scores = []
                    for i in range(self.group_size):
                        for c in cands[i]:
                            cand_scores.append((new_scores[i, c], i, c))

                    if not self.random_sample:
                        if seq.shape[1] == 1:
                            order = (-np.array(cand_scores[
                                :max(len(cands[0]), self.beam_size)])[:, 0]
                            ).argsort().astype(np.int32)
                        else:
                            order = (-np.array(cand_scores)[:, 0]
                                     ).argsort().astype(np.int32)
                    else:
                        count = len(cand_scores)
                        scr = np.array([c[0] for c in cand_scores])
                        p = softmax(scr[np.newaxis])[0]
                        num_sampled = min(2 * self.group_size**2,
                                          np.nan_to_num(p).nonzero()[0].size)
                        order = np.random.choice(count, size=num_sampled,
                                                 replace=False, p=p)

                else:
                    new_scores = new_scores.ravel()

                    if not self.random_sample:
                        if seq.shape[1] == 1:
                            order = (-new_scores[:self.voc_size]).argsort(
                            ).astype(np.int32)
                        else:
                            order = (-new_scores).argsort().astype(np.int32)
                    else:
                        count = new_scores.size
                        p = softmax(new_scores[np.newaxis])[0]
                        num_sampled = min(2 * self.group_size**2,
                                          np.nan_to_num(p).nonzero()[0].size)
                        order = np.random.choice(count, size=num_sampled,
                                                 replace=False, p=p)

                for idx in order:
                    if new_seq.shape[0] == (g + 1) * self.group_size:
                        break

                    if self.whitelist is not None:
                        seq_in_group, word_idx = cand_scores[idx][1:]
                    else:
                        seq_in_group, word_idx = divmod(idx, self.voc_size)

                    extended_seq = np.concatenate(
                        [seq[self.group_size * g + seq_in_group],
                         np.array([word_idx], dtype=np.int32)])

                    if self.whitelist is not None:
                        scr = cand_scores[idx][0]
                    else:
                        scr = new_scores[idx]

                    if extended_seq[-1] == self.w_to_idx['</s>']:
                        finished.append((extended_seq, scr))
                        finished_dec_inits.append(dec_init[seq_in_group])
                    elif scr > -np.inf:
                        if self.whitelist is not None:
                            index_in_group = new_seq.shape[0] % self.group_size
                            k = g * self.group_size + index_in_group
                            trie_positions[k] = go_down_trie(
                                self.whitelist, extended_seq)

                        new_seq = np.vstack([new_seq, extended_seq])
                        new_dec_inits.append(dec_init[seq_in_group])
                        next_scores.append(scr)

                if new_seq.shape[0] < (g + 1) * self.group_size:
                    pad_len = (g + 1) * self.group_size - new_seq.shape[0]
                    new_seq = np.vstack(
                        [new_seq, np.ones((pad_len, new_seq.shape[1]))]
                    ).astype(np.int32)
                    new_dec_inits.append(np.zeros((pad_len, dec_init.shape[1]),
                                                  dtype=np.float32))
                    next_scores += [-np.inf] * pad_len

            if not new_seq.size:
                print('Ending...')
                break

            seq = new_seq
            scores = np.array(next_scores)
            dec_init = np.vstack(new_dec_inits)

            if self.verbose_log:
                print('Length ', seq.shape[1], '\n')
                for utt, s in zip(seq, scores):
                    print('{:.4f} {}'.format(s, print_utt(utt)))
                    print('')
                print('#############\n')

        return finished, finished_dec_inits
