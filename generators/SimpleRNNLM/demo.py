from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import lasagne as L

sys.path.append('../')
from SimpleRNNLN import SimpleRNNLN
from diverse_beam_search import DiverseBeamSearch, softmax
from data_load.mt_load import get_mt_voc


parser = argparse.ArgumentParser(description='HRED demo.')
parser.add_argument('-m', '--model', default=None)
parser.add_argument('-mt', '--mt_path', default='data/mtriples')

args = parser.parse_args()


if args.model is None:
    sys.exit("Please provide a model file: -m path/to/model.npz. Aborting.")

idx_to_w, w_to_idx, voc_size, _ = get_mt_voc(path=args.mt_path)

net = SimpleRNNLN(
    voc_size=voc_size,
    emb_size=300,
    rec_size=300,
    skip_train=True)

net.load_params(args.model)


def print_utt(utt):
    return ' '.join([idx_to_w[x] for x in utt])


def utt_to_array(utt):
    arr = np.array([w_to_idx.get(w, w_to_idx['<unk>']) for w in utt])[
        np.newaxis].astype(np.int32)
    arr[arr == -voc_size] = -1
    return arr


def context_summary(context, lookup=True):
    con_init = np.zeros((1, net.rec_size), dtype=np.float32)
    for utt in context:
        con_init = net.get_new_con_init_fn(
            utt_to_array(utt) if lookup else utt, con_init)
    return con_init

''' Optional whitelist of answers '''
print("Loading whitelist...")
mt = np.load(os.path.join(args.mt_path, 'Training.triples.pkl'))

answers = []
for s in mt:
    answers.append(s[:s.index(2)+1])
answers = answers[:5000]

whitelist = {}
for a in answers:
    dic = whitelist
    for w in a:
        if w not in dic:
            dic[w] = {}
        dic = dic[w]

print("Done")

###################


def talk(
    beam_size=20,
    group_size=2,
    mean=True,
    rank_penalty=0,
    group_diversity_penalty=1,
    seq_diversity_penalty=1,
    short_context=False,
    random=False,
    sharpen_probs=None,
    bs_random=False,
    sharpen_bs_probs=None,
    use_whitelist=False):

    beamsearch = DiverseBeamSearch(
        idx_to_w, net, beam_size, group_size,
        rank_penalty=rank_penalty,
        group_diversity_penalty=group_diversity_penalty,
        seq_diversity_penalty=seq_diversity_penalty,
        unk_penalty=100,
        sharpen_probs=sharpen_bs_probs,
        random_sample=bs_random,
        whitelist=whitelist if use_whitelist else None)

    user_input = sys.stdin.readline()

    context = [('<s> ' + user_input + ' </s>').split()]

    gen_init = net.get_probs_and_new_dec_init_fn(
        utt_to_array(context),
        np.zeros((1, net.rec_size), dtype=np.float32))[1]
    gen_init = np.repeat(gen_init, beam_size, axis=0)

    def len_bonus(size): return 0  # np.log(size)**2

    def fn_score(x, y, mean=mean, len_bonus=len_bonus):
        denom = (x.size - 1) if mean else 1
        return (y + len_bonus(x.size)) / denom

    while True:
        candidates, gen_inits = beamsearch.search(gen_init)

        score_order, inits = zip(*sorted(
            zip(candidates, gen_inits),
            key=lambda ((x1, x2), y): fn_score(x1, x2), reverse=True))
        # alphabetic_order = sorted(
        #     candidates, key=lambda x: ' '.join(print_utt(x[0][1:-1])))

        if not random:
            bot_response = print_utt(score_order[0][0])
            gen_init = inits[0]
        else:
            scr = np.array([[fn_score(x, y) for x, y in score_order]])
            p = softmax(
                scr if sharpen_probs is None else -(-scr)**sharpen_probs)[0]
            index = np.random.choice(len(score_order), p=p)
            bot_response = print_utt(
                score_order[index][0])
            gen_init = inits[index]

        print('######################')
        for x, y in score_order[:10]:
            print('{:.3f}'.format(fn_score(x, y)), '  ', print_utt(x))
        print('######################')

        print(' '.join(bot_response.split()[1:-1]))

        user_input = sys.stdin.readline()
        user_input = ('<s> ' + user_input + ' </s>').split()

        if not short_context:
            gen_init = net.get_probs_and_new_dec_init_fn(
                utt_to_array(user_input),
                np.repeat(gen_init, beam_size, axis=0))[1]
        else:
            context = bot_response.split() + user_input
            get_init = net.get_probs_and_new_dec_init_fn(
                utt_to_array(context),
                np.zeros((1, net.rec_size), dtype=np.float32))[1]

        gen_init = np.repeat(gen_init, beam_size, axis=0)
