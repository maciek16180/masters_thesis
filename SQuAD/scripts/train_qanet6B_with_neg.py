from itertools import chain
import numpy as np
import lasagne as L
import time

import sys
sys.path.append('../')

import QANet


squad_path = '/pio/data/data/squad/glove6B/'
glove_path = '/pio/data/data/glove_vec/6B/glove/'

use_negative_examples = True

if not use_negative_examples:
    print "Only positive samples"
    data = np.load(squad_path + 'train_words.pkl')
    data_char = np.load(squad_path + 'train_char_ascii.pkl')
    train_bin_feats = np.load(squad_path + 'train_bin_feats.pkl')
else:
    print "Using negative samples from wikipedia"
    squad_path += 'wiki_negative_train/'
    data_pos = np.load(squad_path + 'train_pos_words.pkl')
    data_neg = np.load(squad_path + 'train_neg_words.pkl')
    data_char_pos = np.load(squad_path + 'train_pos_char_ascii.pkl')
    data_char_neg = np.load(squad_path + 'train_neg_char_ascii.pkl')
    train_bin_feats_pos = np.load(squad_path + 'train_pos_bin_feats.pkl')
    train_bin_feats_neg = np.load(squad_path + 'train_neg_bin_feats.pkl')

    data = data_pos + data_neg
    data_char = data_char_pos + data_char_neg
    train_bin_feats = train_bin_feats_pos + train_bin_feats_neg

glove_embs = np.load(glove_path + 'glove.6B.300d.npy')
voc_size = glove_embs.shape[0]
alphabet_size = 128

def filter_broken_answers(data, data_char, train_bin_feats):
    data_new = []
    data_char_new = []
    train_bin_feats_new = []
    for i in xrange(len(data)):
        if data[i][0]:
            data_new.append(data[i])
            data_char_new.append(data_char[i])
            train_bin_feats_new.append(train_bin_feats[i])
    return data_new, data_char_new, train_bin_feats_new

def trim_data(data, data_char, train_bin_feats, trim=300):
    data_new = []
    data_char_new = []
    train_bin_feats_new = []
    for i in xrange(len(data)):
        if len(data[i][2]) > trim:
            if data[i][0][0][-1] < trim:
                data_new.append(data[i][:2] + [data[i][2][:trim]])
                data_char_new.append([data_char[i][0], data_char[i][1][:trim]])
                train_bin_feats_new.append(train_bin_feats[i][:trim])
        else:
            data_new.append(data[i])
            data_char_new.append(data_char[i])
            train_bin_feats_new.append(train_bin_feats[i])
    return data_new, data_char_new, train_bin_feats_new

data, data_char, train_bin_feats = filter_broken_answers(data, data_char, train_bin_feats)
data, data_char, train_bin_feats = trim_data(data, data_char, train_bin_feats)

data = (data, data_char, train_bin_feats)

###

net = QANet.QANet(voc_size=voc_size,
                  alphabet_size=alphabet_size,
                  emb_size=300,
                  emb_char_size=20,
                  num_emb_char_filters=200,
                  rec_size=300,
                  emb_init=glove_embs,
                  train_inds=[400001],
                  emb_dropout=True,
                  working_path='../evaluate/glove6B/training/',
                  dev_path='/pio/data/data/squad/glove6B/wiki_negative_dev/')

model_filename = '../trained_models/glove6B/wiki_negative/charemb_trainNAW_dropout/charemb_trainNAW_dropout'

num_epochs = 100

t0 = time.time()
for epoch in xrange(1, num_epochs + 1):
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=data, batch_size=12, log_interval=200)
    print '\nTraining loss:   {}'.format(train_error)
    net.save_params(model_filename + '_ep{}'.format(epoch))

print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Models saved as ' + model_filename
