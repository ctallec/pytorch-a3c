import math
import os
import sys

import numpy
import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_atari_env
from learners import ACLearner
from learners import ArtLearner
from models import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms


def train(agent, rank, args, shared_model, optimizer=None):
    try:
        torch.manual_seed(args.seed + rank)

        env = create_atari_env(args.env_name)
        env.seed(args.seed + rank)
        numpy.random.seed(args.seed + rank)

        model = ActorCritic(env.observation_space.shape[0], env.action_space)

        if optimizer is None:
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

        model.train()

        agent_dict = {'ac':'ACLearner', 'art':'ArtLearner'}
        learner = eval(agent_dict[agent])(env, shared_model, optimizer, args)

        learner.train()
    except KeyboardInterrupt:
        print('\ntrain process #{} interrupted\n'.format(rank))
