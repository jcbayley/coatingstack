import numpy as np
import torch 
from torch.nn import functional as F
from collections import deque
from truncated_normal import TruncatedNormalDist

class ReplayBuffer:
    def __init__(self):
        self.discrete_actions = []
        self.continuous_actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
    
    def clear(self):
        del self.discrete_actions[:]
        del self.continuous_actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]

    def update(
            self, 
            discrete_action, 
            continuous_action, 
            state, 
            logprob, 
            reward, 
            state_value, 
            done):
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.states.append(state)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)

class PPO(object):

    def __init__(
            self, 
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size, 
            lr=1e-4, 
            lower_bound=0.1, 
            upper_bound=1.0,
            clip_ratio=0.5,
            n_updates=3):

        print("sd", state_dim)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.policy = Policy(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size)
        
        self.policy_old = Policy(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.lr = lr
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer()
        self.clip_ratio = clip_ratio
        self.n_updates=n_updates


    def update(self):

        R = 0
        policy_loss = []
        returns = deque()

        eps = 1e-8
        for r in self.replay_buffer.rewards[::-1]:
            R = r + 0.99 * R
            returns.appendleft(R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for _ in range(self.n_updates):
            nstate = np.array(self.replay_buffer.states)[:,0]
            action, actiond, actionc, new_log_prob, d_probs, c_means, c_std, entropy = self.select_action(nstate)

            catlprob = torch.cat(self.replay_buffer.logprobs, dim=0)
            ratios = torch.exp(new_log_prob - catlprob)
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * returns

            policy_loss = (-torch.min(surr1, surr2) - 0.01*entropy).mean()

            self.optimiser.zero_grad()
            policy_loss.backward()
            self.optimiser.step()

        self.replay_buffer.clear()

        self.policy_old.load_state_dict(self.policy.state_dict())

        return policy_loss.item()


    def select_action(self, state):

        state = torch.from_numpy(state).float()
        d_probs, c_means, c_std = self.policy(state)

        #print(d_probs.size(), c_means.size())
        #d_onehot = F.one_hot(d_probs, num_classes=policy.output_dim_discrete)
        d = torch.distributions.Categorical(d_probs)
        actiond = d.sample()

        c = TruncatedNormalDist(c_means, c_std, self.lower_bound, self.upper_bound)
        actionc = c.sample()

        log_prob = d.log_prob(actiond) + c.log_prob(actionc).squeeze()

        entropy = d.entropy() + c._entropy

        #print(d.log_prob(actiond).size() , c.log_prob(actionc).size())
        #print(actionc.size(), actiond.size())
        #print(log_prob.size())
        action = torch.cat([actiond, actionc.squeeze(1)], dim=-1)

        #policy.saved_log_probs.append(log_prob)

        return action, actiond, actionc, log_prob, d_probs, c_means, c_std, entropy


    


class Policy(torch.nn.Module):
    def __init__(self, input_dim, output_dim_discrete, output_dim_continuous, hidden_dim, lower_bound=0, upper_bound=1):
        super(Policy, self).__init__()
        self.output_dim_discrete = output_dim_discrete
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.0)
        self.output_discrete = torch.nn.Linear(hidden_dim, output_dim_discrete)
        self.output_continuous_mean = torch.nn.Linear(hidden_dim, output_dim_continuous)
        self.output_continuous_std = torch.nn.Linear(hidden_dim, output_dim_continuous)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.input(x)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.affine3(x)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.affine4(x)
        #x = self.dropout(x)
        x = F.relu(x)
        action_discrete = self.output_discrete(x)
        action_mean = self.output_continuous_mean(x)
        action_std = self.output_continuous_std(x)
        amean = torch.sigmoid(action_mean)*(self.upper_bound - self.lower_bound) + self.lower_bound
        astd = torch.nn.functional.softplus(action_std)*(self.upper_bound - self.lower_bound)*0.5 + 1e-8
        adisc = F.softmax(action_discrete, dim=-1)
        return adisc, amean, astd




def select_action(policy, state):
    state = torch.from_numpy(state).flatten().float().unsqueeze(0)
    d_probs, c_means, c_std = policy(state)

    #d_onehot = F.one_hot(d_probs, num_classes=policy.output_dim_discrete)
    d = torch.distributions.Categorical(d_probs)
    actiond = d.sample()

    c = TruncatedNormalDist(c_means, c_std, 0.1, 1)
    actionc = c.sample()

    log_prob = d.log_prob(actiond) + c.log_prob(actionc)

    action = torch.cat([actiond, actionc.squeeze(0)], dim=-1)

    policy.saved_log_probs.append(log_prob)
    return action


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