import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from copy import deepcopy

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

class A3CLearner:
    def __init__(self, env, shared_model, optimizer, args):
        self.env = env
        self.model = deepcopy(shared_model)
        self.shared_model = shared_model
        self.optimizer = optimizer

        self.values = []
        self.actions = []
        self.rewards = []
        self.entropies = []

        self.gamma = args.gamma
        self.tau = args.tau
        self.num_steps = args.num_steps
        self.max_episode_length = args.max_episode_length

    def train(self):
        self.model.train()
        state = self.env.reset()
        state = torch.from_numpy(state)
        episode_length = 0
        done = True


        while True:
            episode_length += 1
            self.model.load_state_dict(self.shared_model.state_dict())
            if done:
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            for step in range(self.num_steps):
                value, logit, (hx, cx) = self.model(
                    (Variable(state.unsqueeze(0)), (hx, cx)))
                prob = F.softmax(logit)
                action = prob.multinomial()
                entropy = -(prob * prob.log()).sum(1)
                self.entropies.append(entropy)

                state, reward, done, _ = self.env.step(action.data.numpy())
                done = done or episode_length >= self.max_episode_length
                reward = max(min(reward, 1), -1)

                if done:
                    episode_length = 0
                    state = self.env.reset()

                state = torch.from_numpy(state)
                self.values.append(value)
                self.actions.append(action)
                self.rewards.append(reward)

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                value, _, _ = self.model((Variable(state.unsqueeze(0)), (hx, cx)))
                R = value.data
            self.values.append(Variable(R))

            self.learn()
            
    def learn(self):
        gae = torch.zeros(1, 1)
        R = self.values[-1]

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss += 0.5 * advantage.pow(2)

            # GAE
            delta_t = self.rewards[i] + self.gamma * \
                    self.values[i + 1].data - self.values[i].data
            gae = gae * self.gamma * self.tau + delta_t

            policy_loss -= 0.01 * self.entropies[i]
            self.actions[i].reinforce(gae[0, 0])

        self.optimizer.zero_grad()

        variables = self.actions + [policy_loss + 0.5 * value_loss]
        grad_variables = [None] * len(self.actions) + [torch.ones(1, 1)]
        torch.autograd.backward(variables, grad_variables)
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)

        ensure_shared_grads(self.model, self.shared_model)
        self.optimizer.step()

        self.actions = []
        self.values = []
        self.rewards = []
        self.entropies = []
