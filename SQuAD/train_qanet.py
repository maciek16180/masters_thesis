from itertools import chain
import numpy as np
import lasagne as L
import time
import QANet_glove


squad_path = '/pio/data/data/squad/'
glove_path = '/pio/data/data/glove_vec/6B/'

data = np.load(squad_path + 'train_with_glove_vocab.pkl')
data_char = np.load(squad_path + 'train_char_ascii.pkl')
glove_embs = np.load(glove_path + 'glove.6B.300d.npy')
voc_size = glove_embs.shape[0]
alphabet_size = 128

def filter_broken_answers(data, data_char):
    data_new = []
    data_char_new = []
    for i in xrange(len(data)):
        if data[i][0]:
            data_new.append(data[i])
            data_char_new.append(data_char[i])
    return data_new, data_char_new

data, data_char = filter_broken_answers(data, data_char)
data = (data, data_char)
    
trim = 300
data = zip(*[(d0[:2] + [d0[2][:trim]], [d1[0], d1[1][:trim]]) \
             for d0, d1 in zip(*data) if max(d0[0][0]) < trim])

###

update_fn = lambda l, p: L.updates.adam(l, p)

net = QANet_glove.QANet(voc_size=voc_size,
                        alphabet_size=alphabet_size,
                        emb_size=300,
                        emb_char_size=20,
                        num_emb_char_filters=200,
                        rec_size=300,
                        emb_init=glove_embs,
                        train_inds=[0],
                        update_fn=update_fn)

model_filename = 'trained_models/glove_unks/charemb_glove_train_unk_dropout'

num_epochs = 10

t0 = time.time()
for epoch in xrange(1, num_epochs + 1):
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=data, batch_size=15, log_interval=200)
    print '\nTraining loss:   {}'.format(train_error)
    net.save_params(model_filename + '_ep{}'.format(epoch))
    
print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Models saved as ' + model_filename
