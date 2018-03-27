from __future__ import print_function

import os
import sys
import io
import argparse
import inspect
import numpy as np
import lasagne as L
from datetime import datetime
from nltk.tokenize import regexp_tokenize


sys.path.append('../')
from HRED import HRED
from diverse_beam_search import DiverseBeamSearch, softmax
from data_load.mt_load import get_mt_voc

'''
    Demo for HRED. Run in interactive Python!
    Uses Diverse Beam Search from https://arxiv.org/pdf/1610.02424.pdf

        --model             Path to a npz model file
        --mt_path           Path to MovieTriples data set
        --log               Directory for conversation logs, default 'log'
        --mode              'full' (softmax) or 'ssoft' (sampled softmax)
                                (default 'ssoft'). It has to be the same as in
                                traning.

    Once the model is built, run talk() function for the interactive dialogue
    demo. Arguments (with default values):

        beam_size                 20     Beam size for DBS
        group_size                2      Group size for DBS
        mean                      True   Score for a sentence is an average
                                         from all tokens
        rank_penalty              0
        group_diversity_penalty   1      DBS parameter
        seq_diversity_penalty     1      Penalizes repeating tokens in a single
                                         sentence
        short_context             True   Use only the last two sentences as
                                         a context
        random                    False  Sample from DBS results based on
                                         their softened scores
        sharpen_probs             None   float > 1.0 Sharpens above
                                         probabilities to better differentiate
                                         between stronger and weaker choices
        bs_random                 False  Use random sampling in DBS steps
        use_whitelist             False  Force the model to choose a response
                                         from top 5000 utterances from
                                         MovieTriples
'''

parser = argparse.ArgumentParser(description='HRED demo.')
parser.add_argument('-m', '--model', default=None)
parser.add_argument('-mt', '--mt_path', default='data/mtriples')
parser.add_argument('-l', '--log', default='log')
parser.add_argument('-md', '--mode', default='ssoft')

args = parser.parse_args()


if args.model is None:
    sys.exit("Please provide a model file: -m path/to/model.npz. Aborting.")

if not os.path.exists(args.log):
    os.makedirs(args.log)

idx_to_w, w_to_idx, voc_size, _ = get_mt_voc(path=args.mt_path)

net = HRED(
    voc_size=voc_size,
    emb_size=300,
    lv1_rec_size=300,
    lv2_rec_size=300,
    out_emb_size=300,
    num_sampled=200,
    mode=args.mode,
    skip_train=True)

net.load_params(args.model)

all_params = [x.name for x in L.layers.get_all_params(net.train_net)]
all_params = dict(
    zip(all_params, L.layers.get_all_param_values(net.train_net)))


def print_utt(utt):
    return ' '.join([idx_to_w[x] for x in utt])


def utt_to_array(utt):
    arr = np.array([w_to_idx.get(w, w_to_idx['<unk>']) for w in utt])[
        np.newaxis].astype(np.int32)
    arr[arr == -voc_size] = -1
    return arr


def context_summary(context, lookup=True):
    con_init = np.zeros((1, net.lv2_rec_size), dtype=np.float32)
    for utt in context:
        con_init = net.get_new_con_init_fn(
            utt_to_array(utt) if lookup else utt, con_init)
    return con_init


def tokenize(s):
    if type(s) is not unicode:
        s = s.decode('utf8')
    return regexp_tokenize(s, pattern='[^\W_]+|\S')


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
        seq_diversity_penalty=2,
        short_context=True,
        random=False,
        sharpen_probs=None,
        bs_random=False,
        use_whitelist=False,
        show_candidates=False):

    beamsearch = DiverseBeamSearch(
        idx_to_w, net, beam_size, group_size,
        rank_penalty=rank_penalty,
        group_diversity_penalty=group_diversity_penalty,
        seq_diversity_penalty=seq_diversity_penalty,
        unk_penalty=100,
        random_sample=bs_random,
        whitelist=whitelist if use_whitelist else None)

    sys.stdout.write('ME : ')
    user_input = ['<s>'] + tokenize(sys.stdin.readline().lower()) + ['</s>']

    context = [user_input]
    con_init = context_summary(context, lookup=True)
    W = all_params['dec_init.W']
    b = all_params['dec_init.b']
    dec_init = np.repeat(np.tanh(con_init.dot(W) + b), beam_size, axis=0)

    def len_bonus(size): return 0  # np.log(size)**2

    def fn_score(x, y, mean=mean, len_bonus=len_bonus):
        denom = (x.size - 1) if mean else 1
        return (y + len_bonus(x.size)) / denom

    flogname = os.path.join(args.log, str(datetime.now()))

    with io.open(flogname, 'a', encoding='utf8') as flog:
        flog.write(u"Run params:\n")
        for arg in vars(args):
            s = unicode(arg.ljust(25) + str(getattr(args, arg)), 'utf8')
            flog.write(s + '\n')
        flog.write(u'\n')

        fargs, _, _, values = inspect.getargvalues(inspect.currentframe())
        for argname in fargs:
            flog.write(u'%s = %s\n' % (argname, values[argname]))
        flog.write(u'\n\n######################\n\n')

    while True:
        with io.open(flogname, 'a', encoding='utf8') as flog:
            flog.write(u'ME : ' + u' '.join(user_input[1:-1]) + '\n')

        candidates = beamsearch.search(dec_init)[0]

        order = sorted(
            candidates, key=lambda (x, y): fn_score(x, y), reverse=True)

        if not random:
            bot_response = print_utt(order[0][0])
        else:
            # for x,y in order:
            #     print(x, len(x), print_utt(x), len(print_utt(x)))
            scr = np.array([[fn_score(x, y) if len(print_utt(x[1:-1])) < 100 \
                else -np.inf for x, y in order]])
            # print(scr)
            p = softmax(
                scr if sharpen_probs is None else -(-scr)**sharpen_probs)[0]
            # print(p)
            bot_response = print_utt(
                order[np.random.choice(len(order), p=p)][0])

        log_cands = u'\n######################\n'
        for x, y in order[:10]:
            log_cands += '{:.3f}'.format(fn_score(x, y)) + '  ' + \
                print_utt(x[1:-1]) + '\n'
        log_cands += '######################\n'

        if show_candidates:
            print(log_cands)
        with io.open(flogname, 'a', encoding='utf8') as flog:
            flog.write(log_cands + '\n')

        response = u' '.join(bot_response.split()[1:-1])

        print(u'BOT: ' + response)
        with io.open(flogname, 'a', encoding='utf8') as flog:
            flog.write(u'BOT: ' + response + '\n')

        sys.stdout.write('ME : ')
        user_input = \
            ['<s>'] + tokenize(sys.stdin.readline().lower()) + ['</s>']

        if not short_context:
            con_init = net.get_new_con_init_fn(
                utt_to_array(bot_response), con_init)
            con_init = net.get_new_con_init_fn(
                utt_to_array(user_input), con_init)
        else:
            context = [bot_response.split(), user_input]
            con_init = context_summary(context, lookup=True)

        dec_init = np.repeat(np.tanh(con_init.dot(W) + b), beam_size, axis=0)
