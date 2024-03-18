import numpy as np
import torch 
from torch.nn import functional as F
from collections import deque
from truncated_normal import TruncatedNormalDist

class PO(object):

    def __init__(self, state_dim, num_discrete, num_cont, hidden_size, lr=1e-4):

        print("sd", state_dim)
        self.policy = Policy(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size)
        self.lr = lr
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=self.lr)


    def update(self):

        finish_episode(self.policy, self.optimiser)

    def select_action(self, state, index):

        action, d_probs, c_means, c_std = select_action(self.policy, state, index)

        return action, d_probs, c_means, c_std
    


class Policy(torch.nn.Module):
    def __init__(self, input_dim, output_dim_discrete, output_dim_continuous, hidden_dim):
        super(Policy, self).__init__()
        self.output_dim_discrete = output_dim_discrete
        self.input = torch.nn.Linear(input_dim+1, hidden_dim)
        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = torch.nn.Linear(hidden_dim+2, hidden_dim)
        self.affine4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.0)
        self.output_discrete = torch.nn.Linear(hidden_dim, output_dim_discrete)
        self.output_continuous_mean = torch.nn.Linear(hidden_dim, output_dim_continuous)
        self.output_continuous_std = torch.nn.Linear(hidden_dim, output_dim_continuous)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, index):
        x = torch.cat([x, index], dim=-1)
        x = self.input(x)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        #x = self.dropout(x)
        x = F.relu(x)

        action_discrete = self.output_discrete(x)
        d_probs = F.softmax(action_discrete, dim=-1)
        d = torch.distributions.Categorical(d_probs)
        actiond = d.sample()

        #x = self.dropout(x)
        x = self.affine3(torch.cat([x, actiond.unsqueeze(0), index], dim=-1))
        x = F.relu(x)
        x = self.affine4(x)
        #x = self.dropout(x)
        x = F.relu(x)
        action_mean = self.output_continuous_mean(x)
        action_std = self.output_continuous_std(x)
        return d_probs, torch.sigmoid(action_mean), torch.sigmoid(action_std) + 1e-8, actiond




def select_action(policy, state, index):
    state = torch.from_numpy(state).flatten().float().unsqueeze(0)
    d_probs, c_means, c_std, actiond = policy(state, index)

    #d_onehot = F.one_hot(d_probs, num_classes=policy.output_dim_discrete)
    d = torch.distributions.Categorical(d_probs)
    #actiond = d.sample()

    c = TruncatedNormalDist(c_means, c_std, 0.1, 1)
    actionc = c.sample()

    log_prob = d.log_prob(actiond) + c.log_prob(actionc)

    action = torch.cat([actiond, actionc.squeeze(0)], dim=-1)

    policy.saved_log_probs.append(log_prob)
    return action, d_probs, c_means, c_std


def finish_episode(policy, optimiser):
    R = 0
    policy_loss = []
    returns = deque()

    eps = 1e-8
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        returns.appendleft(R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimiser.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimiser.step()

    policy.rewards = []
    policy.saved_log_probs = []