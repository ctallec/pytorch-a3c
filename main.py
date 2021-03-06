from __future__ import print_function

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from envs import create_atari_env
from train import train
from test import test
from utils import build_model
import my_optim

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='Asynchronous AC and Art')

subparsers = parser.add_subparsers(dest='agent')
subparsers.required = True

ac_parser = subparsers.add_parser('ac', help='actor critic')
art_parser = subparsers.add_parser('art', help='art')

parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PongDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False, metavar='O',
                    help='use an optimizer without shared momentum.')

ac_parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')

art_parser.add_argument('--dicho', action='store_true',
                       help='model decomposes value function dichotomically')
art_parser.add_argument('--remove-constant', action='store_true',
                        help='the value model learns a model of the form c * T'
                        '+ V')
art_subparsers = art_parser.add_subparsers(dest='lambda_type')

constant_art_parser = art_subparsers.add_parser('constant')
decaying_art_parser = art_subparsers.add_parser('decaying')

decaying_art_parser.add_argument('--alpha', default=3, metavar='A',
                                 help='alpha parameter of art')
decaying_art_parser.add_argument('--L0', default=100, metavar='L0',
                                 help='L0 parameter of art')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'  
  
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = create_atari_env(args.env_name)
    shared_model = build_model(env.observation_space.shape[0],
                               env.action_space, args)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(args.agent, rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
    try:
        for p in processes:
                p.join()
    except KeyboardInterrupt:
        print('\nmain thread interrupted\n')
