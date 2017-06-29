import numpy as np
import torch

from learners import ACLearner
from learners.utils import ensure_shared_grads
from torch.autograd import Variable

class ArtLearner(ACLearner):
    def draw_truncation(self):
        X = 0
        T = 0
        while X == 0:
            T += 1
            X = np.random.binomial(1, 1-self.gamma)
        return T
        
    def __init__(self, env, shared_model, optimizer, args):
        super().__init__(env, shared_model, optimizer, args)
        self.num_steps = self.draw_truncation()

    def learn(self):
        gae = torch.zeros(1, 1)
        R = torch.zeros(1, 1)

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(self.rewards))):
            R = R + self.rewards[i]
            advantage = Variable(R) - self.values[i]
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
        self.num_steps = self.draw_truncation()
