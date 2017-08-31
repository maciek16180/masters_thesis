from HRED import HRED

import sys
sys.path.append('../../')

from data_load.mt_load import load_mt, get_mt_voc, get_w2v_embs
from data_load.subtle_load import load_subtle
import numpy as np
import lasagne as L
import time


mt_path = "/pio/data/data/mtriples/"
# mt_path = "/home/maciek/Desktop/mgr/DATA/MovieTriples_Dataset/"

subtle_data = load_subtle(path=mt_path, split=True, trim=200)
train, _, _ = load_mt(mt_path, split=True, trim=None)
idx_to_w, w_to_idx, voc_size, freqs = get_mt_voc(path=mt_path, train_len=len(train))


net = HRED(voc_size=voc_size,
           emb_size=300,
           lv1_rec_size=300,
           lv2_rec_size=300,
           out_emb_size=300,
           num_sampled=200,
           ssoft_probs=freqs)


num_epochs = 4

model_filename = '../temp/pretrained_subtle_GaussInit_300_300_300_300_ssoft200unigr_bs30_cut200.npz'

t0 = time.time()
for epoch in xrange(num_epochs):
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=subtle_data, batch_size=30, log_interval=1000)
    print '\nTraining loss:   {}'.format(train_error)
    net.save_params(model_filename)

print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Model saved as ' + model_filename
