import numpy as np
import os, sys

os.environ["THEANORC"] = "~/.theanorc.cuda"

import time

sys.path.append('../')

from QANet_old2 import QANet
from squad_tools import load_squad_train, filter_empty_answers, trim_data, train_QANet


squad_path = '/pio/data/data/squad/glove6B/'
glove_path = '/pio/data/data/glove_vec/6B/glove/glove.6B.300d.npy'

train_data = load_squad_train(squad_path, negative_path=None)

glove_embs = np.load(glove_path)
voc_size = glove_embs.shape[0]

train_data = filter_empty_answers(train_data)
train_data = trim_data(train_data, 300)

net = QANet(voc_size=voc_size,
            emb_init=glove_embs,
            train_inds=[],
            emb_dropout=True,
            working_path='../evaluate/glove6B/training_old2/',
            prefetch_word_embs=True)

print("prefetch:", net.prefetch_word_embs)

model_filename = '../trained_models/glove6B/old2/old2'

train_QANet(net, train_data, model_filename, batch_size=15)
