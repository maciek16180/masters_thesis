from AnswerBot import AnswerBot
import json, cPickle, argparse
import numpy as np


parser = argparse.ArgumentParser(description='Train script for QANet.')
parser.add_argument('-g', '--glove_version', choices=['6B', '840B'], default='6B')
parser.add_argument('-o', '--output_dir', default='default_dir')
parser.add_argument('--save_preds', action='store_true')
parser.add_argument('-bs', '--batch_size', default=30, type=int)
parser.add_argument('-m' '--model')

args = parser.parse_args()

glove_ver = args.glove_version

# Load data

# wiki_pos
path_wiki_pos = '/pio/data/data/squad/negative_samples/dev.wiki.pos.json'
wiki_pos = [[d[1], d[3]] for d in json.load(open(path_wiki_pos))]

# wiki_neg
path_wiki_neg = '/pio/data/data/squad/negative_samples/dev.wiki.neg.json'
wiki_neg = [d[1:] for d in json.load(open(path_wiki_neg))]

# regular
path_dev = '/pio/data/data/squad/glove' + glove_ver + '/careful_prep/dev.pkl'
dev = [d[1:3] for d in cPickle.load(open(path_dev))]

# randomized
path_dev_rng = '/pio/data/data/squad/negative_samples/dev.squad.random.json'
dev_rng = json.load(open(path_dev_rng))

def iterate_minibatches(data):
    for start_idx in range(0, len(data), args.batch_size):
        excerpt = slice(start_idx, start_idx + args.batch_size)
        yield data[excerpt]

# Build AnswerBot
model_file = np.load(args.model_file)
glove_path = '/pio/data/data/glove_vec/' + glove_ver + '/glove/'
glove_embs = np.load(glove_path + 'glove.' + glove_ver + '.300d.npy')
glove_dict = cPickle.load(open(glove_path + 'glove.' + glove_ver + '.wordlist.pkl'))

abot = AnswerBot(model_file, glove_embs, glove_dict, glove_ver, **kwargs)
