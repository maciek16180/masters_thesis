from SimpleRNNLM import SimpleRNNLM
from mt_load import load_mt, get_mt_voc, get_w2v_embs
import numpy as np
import lasagne as L
import time


mt_path = "/pio/data/data/mtriples/"
# mt_path = "/home/maciek/Desktop/mgr/DATA/MovieTriples_Dataset/"

train, valid, test = load_mt(path=mt_path, split=False, trim=200)
idx_to_w, w_to_idx, voc_size, freqs = get_mt_voc(path=mt_path, train_len=len(train))
word2vec_embs, word2vec_embs_mask = get_w2v_embs(mt_path)


#def update_fn(l, p):
#    return L.updates.adagrad(l, p, learning_rate=.1)
# using ADAM by default

net = SimpleRNNLM(voc_size=voc_size,
                  emb_size=300,
                  rec_size=300,
                  mode='full',
                  #num_sampled=200,
                  emb_init=word2vec_embs,
                  #update_fn=update_fn,
                  #ssoft_probs=freqs,
                  train_emb=True
                 )

#net.load_params('trained_models/pretrained_subtle_GaussInit_300_300_ssoft200unigr_bs50_cut200_nosplit.npz')


last_scores = [np.inf]
max_epochs_wo_improvement = 5
tol = 0.0001
epoch = 1
best_epoch = None

model_filename = 'trained_models/w2vInit_300_300_full_bs50_cut200_nosplit_early5.npz'

t0 = time.time()
while len(last_scores) <= max_epochs_wo_improvement or last_scores[0] > min(last_scores) + tol:
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=train, batch_size=50, log_interval=200)
    val_error = net.validate(val_data=valid, batch_size=25)
    print '\nTraining loss:   {}'.format(train_error)
    print 'Validation loss: {}'.format(val_error)

    if val_error < min(last_scores):
        print '\nSaving model...'
        net.save_params(model_filename)
        print 'Done saving.'
        best_epoch = epoch

    last_scores.append(val_error)

    if len(last_scores) > max_epochs_wo_improvement + 1:
        del last_scores[0]

    epoch += 1
    
test_error = net.validate(val_data=test, batch_size=25)

print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Best model after {} epochs with loss {}'.format(best_epoch, min(last_scores))
print 'Validation set perplexity: {}'.format(np.exp(min(last_scores)))
print 'Model saved as ' + model_filename

print '\nTest loss: {}'.format(test_error)
print 'Test set perplexity: {}'.format(np.exp(test_error))
