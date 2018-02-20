from __future__ import print_function

import sys
import time
import argparse
import os

'''
    Pre-training script for SimpleRNNLM.

        --mt_path           Path to MovieTriples data set
        --output-dir        Output directory, default is 'output.pretrain'
        --batch_size        Default is 30
        --num_epochs        Number of epochs (default 4)
        --samples           Number of targets in sampled softmax (default 200)
        --log_interval      Every log_interval batches a log entry is made
        --mode              'full' (softmax) or 'ssoft' (sampled softmax)
                                (default 'ssoft')
        --learning_rate     Default is 0.0002 (ADAM)

    Models are saved as <output_dir>/model.epXX.npz.
'''

parser = argparse.ArgumentParser(description='Pre-train script for RNNLM.')
parser.add_argument('-mt', '--mt_path', default='data/mtriples')
parser.add_argument('-o', '--output_dir', default='output.pretrain')
parser.add_argument('-bs', '--batch_size', default=30, type=int)
parser.add_argument('-e', '--num_epochs', default=4, type=int)
parser.add_argument('-s', '--samples', default=200, type=int)
parser.add_argument('-li', '--log_interval', default=5000, type=int)
parser.add_argument('-m', '--mode', choices=['ssoft', 'full'], default='ssoft')
parser.add_argument('-lr', '--learning_rate', default=0.0002, type=float)


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
from data_load.mt_load import load_mt, get_mt_voc
from data_load.subtle_load import load_subtle
from training_tools import pretrain

print("\nRun params:")
for arg in vars(args):
    print(arg.ljust(25), getattr(args, arg))
print("\n")

################################

print("Loading data...")

_, valid, _ = load_mt(path=args.mt_path, split=False)
_, _, voc_size, freqs = get_mt_voc(path=args.mt_path)
subtle_data = load_subtle(path=args.mt_path, split=False)

net = SimpleRNNLM(
    voc_size=voc_size,
    emb_size=300,
    rec_size=300,
    num_sampled=args.samples,
    ssoft_probs=freqs,
    mode=args.mode,
    learning_rate=learning_rate,
    skip_gen=True)

pretrain(
    net=net,
    output_path=args.output_dir,
    num_epochs=args.num_epochs,
    train=subtle_data,
    valid=valid,
    bs=args.batch_size,
    log_interval=args.log_interval)