from __future__ import print_function

import numpy as np
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Train script for QANet.')
parser.add_argument('-g', '--glove_version', choices=['6B', '840B'],
                    default='6B')
parser.add_argument('-o', '--output_dir', default='default_dir')
parser.add_argument('--trim', default=300, type=int)
parser.add_argument('--save_preds', action='store_true')
parser.add_argument('-bs', '--batch_size', default=30, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('-cp', '--checkpoint_examples', default=64000, type=int)
parser.add_argument('--squad_subdir', default='careful_prep')
parser.add_argument('--unk', choices=['mean', 'zero', 'train'],
                    default='train')
parser.add_argument('-n', '--negative', nargs='+', default=[])
parser.add_argument('--conv', choices=['full', 'valid'], default='valid')

args = parser.parse_args()

# set paths
squad_base_path = '/pio/data/data/squad'

output_path = os.path.join('../models', args.glove_version, args.output_dir)
squad_path = os.path.join(
    squad_base_path, 'glove' + args.glove_version, args.squad_subdir)
glove_fname = 'glove.' + args.glove_version + '.300d.npy'
glove_path = os.path.join(
    '/pio/data/data/glove_vec', args.glove_version, 'glove', glove_fname)
preds_path = os.path.join(output_path, 'pred') if args.save_preds else None

glove_words_fname = 'glove.' + args.glove_version + '.wordlist.pkl'
glove_words_path = os.path.join(
    '/pio/data/data/glove_vec', args.glove_version, 'glove', glove_words_fname)

if not os.path.exists(output_path):
    os.makedirs(output_path)
elif os.listdir(output_path):
    sys.exit(
        "Chosen output directory already exists and is not empty. Aborting.")
if preds_path is not None and not os.path.exists(preds_path):
    os.makedirs(preds_path)

# redirect all prints to log file
log_path = os.path.join(output_path, 'log')
print("All prints are redirected to", log_path)
log = open(log_path, 'w', buffering=1)
sys.stderr = log
sys.stdout = log

sys.path.append('../')
from QANet import QANet
from squad_tools import load_squad_train, load_squad_dev, \
    filter_empty_answers, trim_data, train_QANet

print("\nRun params:")
for arg in vars(args):
    print(arg.ljust(25), getattr(args, arg))
print("\n")

################################

print("Loading data...")
glove_embs = np.load(glove_path)
voc_size = glove_embs.shape[0]

if args.unk == 'zero':
    glove_embs[0] = 0

glove_words = np.load(glove_words_path)
NAW_token = glove_words.index('<not_a_word>')

train_data = load_squad_train(squad_path, negative_paths=args.negative,
                              NAW_token=NAW_token)
train_data = filter_empty_answers(train_data)
train_data = trim_data(train_data, args.trim)

dev_data = load_squad_dev(squad_base_path, squad_path, NAW_token=NAW_token,
                          lower_raw=args.glove_version == '6B',
                          make_negative=bool(args.negative))

net = QANet(voc_size=voc_size,
            emb_init=glove_embs,
            dev_data=dev_data,
            predictions_path=preds_path,
            train_unk=args.unk == 'train',
            negative=bool(args.negative),
            init_lrate=args.learning_rate,
            checkpoint_examples=args.checkpoint_examples,
            conv=args.conv)

model_filename = os.path.join(output_path, 'model')

train_QANet(net, train_data, model_filename, batch_size=args.batch_size)
