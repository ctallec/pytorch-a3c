import json
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
from utils import build_model
from collections import deque


def test(rank, args, shared_model):
    try:
        torch.manual_seed(args.seed + rank)

        env = create_atari_env(args.env_name)
        env.seed(args.seed + rank)

        model = build_model(env.observation_space.shape[0], env.action_space,
                            args)

        model.eval()

        state = env.reset()
        state = torch.from_numpy(state)
        reward_sum = 0
        done = True

        start_time = time.time()
        log_stream = open(os.path.join('logs', json.dumps(vars(args),
                                                          separators=(',',':'))), 'w')

        # a quick hack to prevent the agent from stucking
        actions = deque(maxlen=100)
        episode_length = 0
        while True:
            episode_length += 1
            # Sync with the shared model
            if done:
                model.load_state_dict(shared_model.state_dict())
                cx = Variable(torch.zeros(1, 256), volatile=True)
                hx = Variable(torch.zeros(1, 256), volatile=True)
            else:
                cx = Variable(cx.data, volatile=True)
                hx = Variable(hx.data, volatile=True)

            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
            prob = F.softmax(logit)
            action = prob.max(1)[1].data.numpy()

            state, reward, done, _ = env.step(action[0, 0])
            done = done or episode_length >= args.max_episode_length
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
                done = True

            if done:
                print("Time {}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, episode_length))

                log_stream.write('{} {}\n'.format(reward_sum, episode_length))
                log_stream.flush()

                reward_sum = 0
                episode_length = 0
                actions.clear()
                state = env.reset()
                time.sleep(60)

            state = torch.from_numpy(state)
    except KeyboardInterrupt:
        print('\ntest process #{} interrupted\n'.format(rank))
        log_stream.close()
