import numpy as np
import lasagne as L
import sys
sys.path.append('../')

from HRED import HRED
from diverse_beam_search import diverse_beam_search, softmax
from data_load.mt_load import load_mt, get_mt_voc, get_w2v_embs


mt_path = "/pio/data/data/mtriples/"
idx_to_w, w_to_idx, voc_size, _ = get_mt_voc(path=mt_path, train_len=0)

hred_net = HRED(voc_size=voc_size,
                emb_size=300,
                lv1_rec_size=300,
                lv2_rec_size=300,
                out_emb_size=300,
                num_sampled=200)

#hred_net.load_params('../trained_models/pretrained_subtle_GaussInit_300_300_300_300_ssoft200unigr_bs30_cut200.npz')
hred_net.load_params('../trained_models/subtleFixed_300_300_300_300_ssoft200unigr_bs30_cut200_early5.npz')

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

def talk(beam_size=20, group_size=2, mean=True, rank_penalty=0, group_diversity_penalty=1, seq_diversity_penalty=1,
         short_context=False, random=False, sharpen_probs=None , bs_random=False, sharpen_bs_probs=None,
         only_last_groups=False):

    user_input = sys.stdin.readline()

    context = [('<s> ' + user_input + ' </s>').split()]
    con_init = context_summary(context, lookup=True)
    W = L.layers.get_all_param_values(hred_net.train_net)[31]
    b = L.layers.get_all_param_values(hred_net.train_net)[32]
    dec_init = np.repeat(np.tanh(con_init.dot(W) + b), beam_size, axis=0)

    len_bonus = lambda size: 0 #np.log(size)**2

    def fn_score(x, y, mean=mean, len_bonus=len_bonus):
        denom = (x.size - 1) if mean else 1
        return (y + len_bonus(x.size)) / denom

    while True:
        beamsearch = diverse_beam_search(beam_size, group_size, dec_init, voc_size, hred_net,
                                         init_seq=utt_to_array('<s> '.split()), rank_penalty=rank_penalty,
                                         group_diversity_penalty=group_diversity_penalty,
                                         seq_diversity_penalty=seq_diversity_penalty, verbose_log=False,
                                         sample=bs_random, sharpen_probs=sharpen_bs_probs, only_last_groups=only_last_groups)

        score_order = sorted(beamsearch, key=lambda (x,y): fn_score(x, y), reverse=True)
    #     alphabetic_order = sorted(beamsearch, key=lambda x: ' '.join(print_utt(x[0][1:-1])))

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
            context = [bot_response.split(), user_input]
            con_init = context_summary(context, lookup=True)

        dec_init = np.repeat(np.tanh(con_init.dot(W) + b), beam_size, axis=0)
