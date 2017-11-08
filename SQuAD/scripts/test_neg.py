from __future__ import print_function

import json, cPickle, argparse, os, sys
import numpy as np


parser = argparse.ArgumentParser(description='Train script for QANet.')
parser.add_argument('-g', '--glove_version', choices=['6B', '840B'], default='6B')
parser.add_argument('--save_preds', action='store_true')
parser.add_argument('-bs', '--batch_size', default=10, type=int)
parser.add_argument('-m', '--model')
parser.add_argument('--conv', choices=['full', 'valid'], default='valid')
parser.add_argument('--unk', default='train', choices=['mean', 'zero', 'train'])

args = parser.parse_args()

output_dir = os.path.dirname(args.model)

log_path = os.path.join(output_dir, 'neg_test')
print("All prints are redirected to", log_path)
log = open(log_path, 'w', buffering=1)
sys.stderr = log
sys.stdout = log

from AnswerBot import AnswerBot
from AnswerBot import not_a_word_Str as NAW_tok

print("\nRun params:")
for arg in vars(args):
    print(arg.ljust(25), getattr(args, arg))
print("\n")

def iterate_minibatches(data):
    for start_idx in range(0, len(data), args.batch_size):
        excerpt = slice(start_idx, start_idx + args.batch_size)
        yield zip(*data[excerpt])

def calc_naws(data):
    naws = 0
    for batch in iterate_minibatches(data):
        qs, xs = batch
        ans = [a[0] for a in abot.get_answers(qs, xs)]
        naws += len([a for a in ans if NAW_tok in a])
    return naws

glove_ver = args.glove_version

# Build AnswerBot
glove_path = '/pio/data/data/glove_vec/' + glove_ver + '/glove/'
glove_embs = np.load(glove_path + 'glove.' + glove_ver + '.300d.npy')
if args.unk == 'zero':
    glove_embs[0] = 0
glove_dict = cPickle.load(open(glove_path + 'glove.' + glove_ver + '.wordlist.pkl'))

abot = AnswerBot(args.model, glove_embs, glove_dict, glove_ver,
    train_unk=args.unk=='train',
    negative=True,
    conv=args.conv)

print('\nDiscarded paragraphs:\n')

# wiki_pos
path_wiki_pos = '/pio/data/data/squad/negative_samples/dev.wiki.pos.json'
wiki_pos = [[d[1], d[3]] for d in json.load(open(path_wiki_pos))]
naws_wiki_pos = calc_naws(wiki_pos)
print('wiki pos: %.2f' % (float(naws_wiki_pos) / len(wiki_pos)))

# wiki_neg
path_wiki_neg = '/pio/data/data/squad/negative_samples/dev.wiki.neg.json'
wiki_neg = [d[1:] for d in json.load(open(path_wiki_neg))]
naws_wiki_neg = calc_naws(wiki_neg)
print('wiki neg: %.2f' % (float(naws_wiki_neg) / len(wiki_neg)))

# regular
path_dev = '/pio/data/data/squad/glove' + glove_ver + '/careful_prep/dev.pkl'
dev = [d[1:3] for d in cPickle.load(open(path_dev))]
naws_dev = calc_naws(dev)
print('dev     : %.2f' % (float(naws_dev) / len(dev)))

# randomized
path_dev_rng = '/pio/data/data/squad/negative_samples/dev.squad.random.json'
dev_rng = json.load(open(path_dev_rng))
naws_dev_rng = calc_naws(dev_rng)
print('dev rng : %.2f' % (float(naws_dev_rng) / len(dev_rng)))
