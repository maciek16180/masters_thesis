import numpy as np
import lasagne as L
import time

import sys
sys.path.append('../')
from HRED import HRED

from reddit_load import load_pairs, get_reddit_voc


reddit_path = "/pio/data/data/reddit_sample/"

train_pairs, test_pairs = load_pairs(path=reddit_path)
idx_to_w, w_to_idx, voc_size, freqs = get_reddit_voc(path=reddit_path)

###

net = HRED(voc_size=voc_size,
           emb_size=300,
           lv1_rec_size=300, 
           lv2_rec_size=300, 
           out_emb_size=300, 
           num_sampled=200,
           ssoft_probs=freqs,
           n=2)

net.load_params('reddit_singles_randomInit_bs60_early5.npz')

last_scores = [np.inf]
max_epochs_wo_improvement = 5
tol = 0.001
epoch = 1
best_epoch = None

model_filename = 'reddit_pairs_singlesInit_bs60_early5.npz'

t0 = time.time()
while len(last_scores) <= max_epochs_wo_improvement or last_scores[0] > min(last_scores) + tol:
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=train_pairs, batch_size=60, log_interval=500)
    val_error = net.validate(val_data=test_pairs, batch_size=60)
    print '\nTraining loss:   {}'.format(train_error)
    print 'Validation loss: {}'.format(val_error)

    if val_error < min(last_scores):
        print '\nSaving model...'
        net.save_params(model_filename)
        print 'Done saving.'
        best_epoch = epoch

    last_scores.append(val_error)

    if len(last_scores) > max_epochs_wo_improvement+1:
        del last_scores[0]

    epoch += 1


print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Best model after {} epochs with loss {}'.format(best_epoch, min(last_scores))
print 'Validation set perplexity: {}'.format(np.exp(min(last_scores)))
print 'Model saved as ' + model_filename
