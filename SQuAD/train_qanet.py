from HRED import HRED
from squad_load import get_glove_train_embs, get_squad_train_voc, \
                        load_squad_train, get_squad_train_chars
from itertools import chain
import numpy as np
import lasagne as L
import time
import QANet


squad_path = '/pio/data/data/squad/'
glove_path = '/pio/data/data/glove_vec/6B/'

data = load_squad_train(squad_path, with_chars=True)
i_to_w, w_to_i, voc_size = get_squad_train_voc(squad_path)
i_to_c, c_to_i, alphabet_size = get_squad_train_chars(squad_path)
glove_embs = get_glove_train_embs(squad_path, glove_path)

def filter_broken_answers(data):
    return zip(*[d for d in zip(*data) if d[0][0]])

data = filter_broken_answers(data)
data = [map(list, data[0]), map(list, data[1])]

for i in xrange(len(data[0])):
    data[0][i].append(list(chain(*data[0][i][1][1:])))
    data[0][i][1] = data[0][i][1][0]
    
trim = 300
data = zip(*[(d0[:2] + [d0[2][:trim]], [d1[0], d1[1][:trim]]) \
             for d0, d1 in zip(*data) if max(d0[0][0]) < trim])

data = [map(list, data[0]), map(list, data[1])]


qa_net = QANet.QANet(voc_size=voc_size,
                     alphabet_size=alphabet_size,
                     emb_size=300,
                     emb_char_size=100,
                     num_emb_char_filters=200,
                     rec_size=300,
                     emb_init=glove_embs)

print 'Starting training after 4 epochs\n\n'

net.load_params('test_params_char_emb_4ep.npz')
model_filename = 'test_params_char_emb'

num_epochs = 10

t0 = time.time()
for epoch in xrange(num_epochs):
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=data, batch_size=15, log_interval=200)
    print '\nTraining loss:   {}'.format(train_error)
    net.save_params(model_filename)
    
print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Model saved as ' + model_filename