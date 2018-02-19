from __future__ import print_function

import sys
import time
import argparse
import os
import numpy as np


parser = argparse.ArgumentParser(description='Train script for HRED.')
parser.add_argument('-mt', '--mt_path', default='data/mtriples')
parser.add_argument('-o', '--output_dir', default='output')
parser.add_argument('-p', '--pretrained_model', default=None)
parser.add_argument('-bs', '--batch_size', default=30, type=int)
parser.add_argument('-s', '--samples', default=200, type=int)
parser.add_argument('-li', '--log_interval', default=5000, type=int)
parser.add_argument('-m', '--mode', choices=['ssoft', 'full'], default='ssoft')
parser.add_argument('-e', '--emb_init', choices=['random', 'w2v'],
                    default='random')
parser.add_argument('-lr', '--learning_rate', default=0.0002, type=float)
parser.add_argument('--fix_emb', action='store_false')


args = parser.parse_args()

# set paths
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
elif os.listdir(args.output_dir):
    sys.exit(
        "Chosen output directory already exists and is not empty. Aborting.")

# redirect all prints to log file
log_path = os.path.join(args.output_dir, 'log')
print("All prints are redirected to", log_path)
log = open(log_path, 'w', buffering=1)
sys.stderr = log
sys.stdout = log

sys.path.append('../../')
from HRED import HRED
from data_load.mt_load import load_mt, get_mt_voc, get_w2v_embs
from training_tools import train

print("\nRun params:")
for arg in vars(args):
    print(arg.ljust(25), getattr(args, arg))
print("\n")

################################

print("Loading data...")

train, valid, test = load_mt(path=args.mt_path, split=True)
_, _, voc_size, freqs = get_mt_voc(path=args.mt_path)

if args.emb_init == 'w2v':
    emb_init, _ = get_w2v_embs(args.mt_path)
    train_inds = [0, 1, 2]
else:
    emb_init = None
    train_inds = []

net = HRED(
    voc_size=voc_size,
    emb_size=300,
    lv1_rec_size=300,
    lv2_rec_size=300,
    out_emb_size=300,
    num_sampled=args.samples,
    ssoft_probs=freqs,
    mode=args.mode,
    learning_rate=args.learning_rate,
    emb_init=emb_init,
    train_emb=not args.fix_emb,
    train_inds=train_inds,
    skip_gen=True)

if args.pretrained_model is not None:
    net.load_params(args.pretrained_model)

def train(
    net=net,
    output_path=args.output_dir,
    train=train,
    valid=valid,
    test=test,
    bs=args.batch_size,
    log_interval=args.log_interval)
