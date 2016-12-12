from SimpleRNNLM import SimpleRNNLM
from mt_load import load_mt, get_mt_voc, get_w2v_embs
import numpy as np


mt_path = "/pio/data/data/mtriples/"

train, valid, test = load_mt(path=mt_path, split=False, trim=200)
idx_to_w, w_to_idx, voc_size, freqs = get_mt_voc(path=mt_path, train_len=len(train))
word2vec_embs, word2vec_embs_mask = get_w2v_embs(mt_path)


net = SimpleRNNLM(voc_size=voc_size,
                  emb_size=300,
                  rec_size=300,
                  mode='ssoft',
                  num_sampled=200,
                  emb_init=word2vec_embs,
                  ssoft_probs=freqs)


last_scores = [np.inf]
max_epochs_wo_improvement = 5
tol = 0.001
epoch = 1

model_filename = 'w2vInit_300_300_ssoft200unigr_bs50_cut200_nosplit_early5.npz'

while len(last_scores) <= max_epochs_wo_improvement or last_scores[0] > min(last_scores) + tol:
    print '\n\nStarting epoch {}...\n'.format(epoch)
    net.train_one_epoch(train_data=train, batch_size=50, log_interval=200)
    val_error = net.validate(val_data=valid, batch_size=25)

    if val_error < min(last_scores):
        print 'Saving model...'
        net.save_params(model_filename)
        print 'Done saving.'

    last_scores.append(val_error)

    if len(last_scores) > max_epochs_wo_improvement+1:
        del last_scores[0]

    epoch += 1
