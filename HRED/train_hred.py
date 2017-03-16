from HRED import HRED
from mt_load import load_mt, get_mt_voc, get_w2v_embs
import numpy as np
import lasagne as L
import time


mt_path = "/pio/data/data/mtriples/"
# mt_path = "/home/maciek/Desktop/mgr/DATA/MovieTriples_Dataset/"

train, valid, test = load_mt(path=mt_path, split=True, trim=200)
idx_to_w, w_to_idx, voc_size, freqs = get_mt_voc(path=mt_path, train_len=len(train))
word2vec_embs, word2vec_embs_mask = get_w2v_embs(mt_path)

###
emb_init = word2vec_embs
train_emb = True
###

net = HRED(voc_size=voc_size,
           emb_size=300,
           lv1_rec_size=300, 
           lv2_rec_size=300, 
           out_emb_size=300, 
           num_sampled=200,
           ssoft_probs=freqs,
           emb_init=emb_init,
           train_emb=train_emb)


#net.load_params('pretrained_subtle_GaussInit_300_300_300_300_ssoft200unigr_bs30_cut200.npz')


last_scores = [np.inf]
max_epochs_wo_improvement = 5
tol = 0.001
epoch = 1
best_epoch = None

model_filename = 'w2vInit_300_300_300_300_ssoft200unigr_bs30_cut200_early5.npz'

t0 = time.time()
while len(last_scores) <= max_epochs_wo_improvement or last_scores[0] > min(last_scores) + tol:
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=train, batch_size=30, log_interval=200)
    val_error = net.validate(val_data=valid, batch_size=30)
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

test_error = net.validate(val_data=test, batch_size=30)

print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Best model after {} epochs with loss {}'.format(best_epoch, min(last_scores))
print 'Validation set perplexity: {}'.format(np.exp(min(last_scores)))
print 'Model saved as ' + model_filename

print '\nTest loss: {}'.format(test_error)
print 'Test set perplexity: {}'.format(np.exp(test_error))
