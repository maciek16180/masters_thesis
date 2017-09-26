import numpy as np
import lasagne as L
import sys
sys.path.append('../')
sys.path.append('../../')

from HRED import HRED
from diverse_beam_search import DiverseBeamSearch, softmax
from data_load.redditv3_load import load_pairs, get_redditv3_voc

redditv3_path = "/pio/data/data/reddit_sample/v3/"

train_pairs, test_pairs = load_pairs(path=redditv3_path)
idx_to_w, w_to_idx, voc_size, _ = get_redditv3_voc(path=redditv3_path, train_len=len(train_pairs) * 2)

###

hred_net = HRED(voc_size=voc_size,
                emb_size=300,
                lv1_rec_size=300,
                lv2_rec_size=300,
                out_emb_size=300,
                num_sampled=1000,
                train_emb=False,
                skip_train=True)

subtle_params = np.load('trained_models/subtle_fixed_pretrain/pretrained_subtle_fixed_GaussInit_300_300_300_300_ssoft200unigr_bs30_cut200_ep10.npz')
emb = subtle_params['arr_0']
hred_net.load_params('trained_models/with_subtle/redditv3_pairs_subtleFixed_bs100_early5_ep25.npz', emb)

def print_utt(utt):
    return ' '.join([idx_to_w[x] for x in utt])

def utt_to_array(utt):
    arr = np.array([w_to_idx.get(w, w_to_idx['<unk>']) for w in utt])[np.newaxis].astype(np.int32)
    arr[arr == -voc_size] = -1
    return arr

def context_summary(context, lookup=True):
    con_init = np.zeros((1, hred_net.lv2_rec_size), dtype=np.float32)
    for utt in context:
        con_init = hred_net.get_new_con_init_fn(utt_to_array(utt) if lookup else utt, con_init)
    return con_init

###################

def go_down_trie(trie, seq):
    for x in seq:
        if x not in trie:
            raise KeyError("Sequence is not in trie.")
        trie = trie[x]
    return trie

print "Loading whitelist..."

answers = [s[1] for s in train_pairs if 0 not in s[1] and len(s[1]) > 5]
answers = answers[:5000]

whitelist = {}

for a in answers:
    dic = whitelist
    for w in a:
        if w not in dic:
            dic[w] = {}
        dic = dic[w]

print "Done"

###################

def talk(beam_size=20, group_size=2, mean=True, rank_penalty=0, group_diversity_penalty=1, seq_diversity_penalty=1,
         short_context=False, random=False, sharpen_probs=None , bs_random=False, sharpen_bs_probs=None):

    beamsearch = DiverseBeamSearch(idx_to_w, hred_net, beam_size, group_size,
                                   rank_penalty=rank_penalty,
                                   group_diversity_penalty=group_diversity_penalty,
                                   seq_diversity_penalty=seq_diversity_penalty,
                                   unk_penalty=100,
                                   sharpen_probs=sharpen_bs_probs,
                                   random_sample=bs_random,
                                   whitelist=whitelist)

    user_input = sys.stdin.readline()

    context = [('<s> ' + user_input + ' </s>').split()]
    con_init = context_summary(context, lookup=True)

    param_names = [x.name for x in L.layers.get_all_params(hred_net.train_net)]

    W = L.layers.get_all_param_values(hred_net.train_net)[param_names.index('dec_init.W')]
    b = L.layers.get_all_param_values(hred_net.train_net)[param_names.index('dec_init.b')]

    dec_init = np.repeat(np.tanh(con_init.dot(W) + b), beam_size, axis=0)

    len_bonus = lambda size: 0 #np.log(size)**2

    def fn_score(x, y, mean=mean, len_bonus=len_bonus):
        denom = (x.size - 1) if mean else 1
        return (y + len_bonus(x.size)) / denom

    while True:
        candidates = beamsearch.search(dec_init)

        score_order = sorted(candidates, key=lambda (x,y): fn_score(x, y), reverse=True)
    #     alphabetic_order = sorted(candidates, key=lambda x: ' '.join(print_utt(x[0][1:-1])))

        if not random:
            bot_response = print_utt(score_order[0][0])
        else:
            scr = np.array([[fn_score(x, y) for x, y in score_order]])
            p = softmax(scr if sharpen_probs is None else -(-scr)**sharpen_probs)[0]
            bot_response = print_utt(score_order[np.random.choice(len(score_order), p=p)][0])

        print '######################'
        for x, y in score_order[:10]:
            print '{:.3f}'.format(fn_score(x, y)), '  ', print_utt(x)
        print '######################'

        print ' '.join(bot_response.split()[1:-1])

        user_input = sys.stdin.readline()
        user_input = ('<s> ' + user_input + ' </s>').split()

        if not short_context:
            con_init = hred_net.get_new_con_init_fn(utt_to_array(bot_response), con_init)
            con_init = hred_net.get_new_con_init_fn(utt_to_array(user_input), con_init)
        else:
            context = [user_input]
            con_init = context_summary(context, lookup=True)

        dec_init = np.repeat(np.tanh(con_init.dot(W) + b), beam_size, axis=0)

