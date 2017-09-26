import numpy as np
import lasagne as L
import time

import sys
sys.path.append('../../')
sys.path.append('../../../')
from HRED import HRED

from data_load.redditv3_load import load_pairs, get_redditv3_voc

redditv3_path = "/pio/data/data/reddit_sample/v3/"

train_pairs, test_pairs = load_pairs(path=redditv3_path)
_, _, voc_size, freqs = get_redditv3_voc(path=redditv3_path, train_len=len(train_pairs) * 2) # x2, bo train leci parami

###

net = HRED(voc_size=voc_size,
           emb_size=300,
           lv1_rec_size=300,
           lv2_rec_size=300,
           out_emb_size=300,
           num_sampled=200,
           train_emb=False,
           ssoft_probs=freqs,
           skip_gen=True)

net.load_params('../trained_models/subtle_fixed_pretrain/pretrained_subtle_fixed_GaussInit_300_300_300_300_ssoft200unigr_bs30_cut200_ep10.npz')

last_scores = [np.inf]
max_epochs_wo_improvement = 5
tol = 0.001
epoch = 1
best_epoch = None

model_filename = '../trained_models/with_subtle/redditv3_pairs_subtleFixed_bs100_early5' # dobrac batch size

t0 = time.time()
while len(last_scores) <= max_epochs_wo_improvement or last_scores[0] > min(last_scores) + tol:
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=train_pairs, batch_size=100, log_interval=200)
    print 'validating...'
    val_error = net.validate(val_data=test_pairs, batch_size=100, log_interval=500)
    print '\nTraining loss:   {}'.format(train_error)
    print 'Validation loss: {}'.format(val_error)

    if val_error < min(last_scores):
        print '\nSaving model...'
        net.save_params(model_filename + '_ep' + str(epoch) + '.npz')
        print 'Done saving.'
        best_epoch = epoch

    last_scores.append(val_error)

    if len(last_scores) > max_epochs_wo_improvement + 1:
        del last_scores[0]

    epoch += 1


print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Best model after {} epochs with loss {}'.format(best_epoch, min(last_scores))
print 'Validation set perplexity: {}'.format(np.exp(min(last_scores)))
print 'Model saved as ' + model_filename
