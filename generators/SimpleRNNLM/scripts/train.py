from __future__ import print_function

import sys
import time
import argparse
import os
import numpy as np


parser = argparse.ArgumentParser(description='Train script for RNNLM.')
parser.add_argument('-mt', '--mt_path', default='data/mtriples')
parser.add_argument('-o', '--output_dir', default='output')
parser.add_argument('-p', '--pretrained_model', default=None)
parser.add_argument('-bs', '--batch_size', default=30, type=int)
parser.add_argument('-s', '--samples', default=200, type=int)
parser.add_argument('-li', '--log_interval', default=5000, type=int)
parser.add_argument('-m', '--mode', choices=['ssoft', 'full'], default='ssoft')
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
from SimpleRNNLM import SimpleRNNLM
from data_load.mt_load import load_mt, get_mt_voc, get_w2v_embs
from training_tools import train as train_fn

print("\nRun params:")
for arg in vars(args):
    print(arg.ljust(25), getattr(args, arg))
print("\n")

################################

print("Loading data...")

train, valid, test = load_mt(path=args.mt_path, split=False)
_, _, voc_size, freqs = get_mt_voc(path=args.mt_path)

net = SimpleRNNLM(
    voc_size=voc_size,
    emb_size=300,
    rec_size=300,
    num_sampled=args.samples,
    ssoft_probs=freqs,
    mode=args.mode,
    learning_rate=args.learning_rate,
    train_emb=not args.fix_emb,
    skip_gen=True)

if args.pretrained_model is not None:
    net.load_params(args.pretrained_model)

train_fn(
    net=net,
    output_path=args.output_dir,
    train=train,
    valid=valid,
    test=test,
    bs=args.batch_size,
    log_interval=args.log_interval)
