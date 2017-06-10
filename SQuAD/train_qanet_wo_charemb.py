from squad_load import get_glove_train_embs, get_squad_train_voc, \
                        load_squad_train, get_squad_train_chars
from itertools import chain
import numpy as np
import lasagne as L
import time
import QANet_wo_charemb


squad_path = '/pio/data/data/squad/'
glove_path = '/pio/data/data/glove_vec/6B/'

data = np.load(squad_path + 'train_with_unks.pkl')
i_to_w, w_to_i, voc_size = get_squad_train_voc(squad_path)
glove_embs = get_glove_train_embs(squad_path, glove_path)

# Some answers get broken in the process of tokenization, because some answer words are not properly split.
def filter_broken_answers(data):
    return [d for d in data if d[0]]
data = filter_broken_answers(data)

# Originally contexts are split into sentences, this reverses that.
for i in xrange(len(data)):
    data[i].append(list(chain(*data[i][1][1:])))
    data[i][1] = data[i][1][0]
    
trim = 300
data = [d[:2] + [d[2][:trim]] for d in data if max(d[0][0]) < trim]

###

update_fn = lambda l, p: L.updates.adam(l, p)

net = QANet_wo_charemb.QANet(voc_size=voc_size,
                             emb_size=300,
                             rec_size=300,
                             emb_init=glove_embs,
                             update_fn=update_fn)

#et.load_params('')
model_filename = 'trained_models/simplified_unks/simplified_unk'

num_epochs = 8

t0 = time.time()
for epoch in xrange(1, num_epochs + 1):
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=data, batch_size=20, log_interval=200)
    print '\nTraining loss:   {}'.format(train_error)
    net.save_params(model_filename + '_ep{}'.format(epoch))
    
print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Models saved as ' + model_filename
