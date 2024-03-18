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
        self.entropy = []
        self.returns = []

        self.t_discrete_actions = []
        self.t_continuous_actions = []
        self.t_states = []
        self.t_logprobs = []
        self.t_rewards = []
        self.t_state_values = []
        self.t_dones = []
        self.t_entropy = []
    
    def clear(self):
        del self.discrete_actions[:]
        del self.continuous_actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]
        del self.entropy[:]
        del self.returns[:]

    def t_clear(self):
        del self.t_discrete_actions[:]
        del self.t_continuous_actions[:]
        del self.t_states[:]
        del self.t_logprobs[:]
        del self.t_rewards[:]
        del self.t_state_values[:]
        del self.t_dones[:]
        del self.t_entropy[:]

    def update(
            self, 
            discrete_action, 
            continuous_action, 
            state, 
            logprob, 
            reward, 
            state_value, 
            done,
            entropy):
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.states.append(state)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)
        self.entropy.append(entropy)

    def update_returns(self, returns):
        self.returns.extend(returns)

    def next_episode(self, ):
        self.discrete_actions.append(self.t_discrete_actions)
        self.continuous_actions.append(self.t_continuous_actions)
        self.states.append(self.t_states)
        self.logprobs.append(self.t_logprobs)
        self.rewards.append(self.t_rewards)
        self.state_values.append(self.t_state_values)
        self.dones.append(self.t_dones)
        self.entropy.append(self.t_entropy)

        self.t_clear()


class PO(object):

    def __init__(self, state_dim, num_discrete, num_cont, hidden_size, lr_policy=1e-4, lr_value=2e-4, lower_bound=0.1, upper_bound=1.0, n_updates=1):

        print("sd", state_dim)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.policy = Policy(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound)
        
        self.value = Value(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound)
        
        self.lr_value = lr_value
        self.lr_policy = lr_policy
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        self.optimiser_value = torch.optim.Adam(self.value.parameters(), lr=self.lr_value)

        self.mse_loss = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

        self.n_updates = n_updates

    def get_returns(self, rewards):

        temp_r = deque()
        R=0
        for r in rewards[::-1]:
            R = r + 0.99 * R
            temp_r.appendleft(R)
        
        return np.array(temp_r)


    def update(self, update_policy=True):

        R = 0
        policy_loss = []

        eps = 1e-8

        returns = torch.from_numpy(np.array(self.replay_buffer.returns))
        returns = (returns - returns.mean()) / (returns.std() + eps)

        #print(self.replay_buffer.state_values)
        state_vals = torch.cat(self.replay_buffer.state_values).to(torch.float32)
        advantage = returns.detach() - state_vals.detach()

        #for log_prob, R in zip(self.replay_buffer.logprobs, returns):
        #    policy_loss.append(-log_prob * R)
        #policy_loss = torch.cat(policy_loss).mean()
        #print(torch.cat(self.replay_buffer.logprobs).size(), returns.size())
        policy_loss = (-torch.cat(self.replay_buffer.logprobs).squeeze() * advantage.squeeze() - 0.001*torch.cat(self.replay_buffer.entropy).squeeze()).mean()
        #policy_loss = (-torch.cat(self.replay_buffer.logprobs).squeeze() * advantage.squeeze() ).mean()

        
        #print(returns.size(), state_vals.size())
        value_loss = self.mse_loss(returns.to(torch.float32).squeeze(), state_vals.squeeze())

        if update_policy:
            self.optimiser.zero_grad()
            policy_loss.backward()
            self.optimiser.step()

        self.optimiser_value.zero_grad()
        value_loss.backward()
        self.optimiser_value.step()

        self.replay_buffer.clear()

        return policy_loss.item(), value_loss.item()


    def select_action(self, state):

        state = torch.from_numpy(state).flatten().float().unsqueeze(0)
        d_probs, c_means, c_std = self.policy(state)

        state_value = self.value(state)
        #d_onehot = F.one_hot(d_probs, num_classes=policy.output_dim_discrete)
        d = torch.distributions.Categorical(d_probs)
        actiond = d.sample()

        c = TruncatedNormalDist(c_means, c_std, self.lower_bound, self.upper_bound)
        actionc = c.sample()

        log_prob = d.log_prob(actiond) + c.log_prob(actionc)

        action = torch.cat([actiond, actionc.squeeze(0)], dim=-1)

        entropy = d.entropy() + c._entropy

        #policy.saved_log_probs.append(log_prob)

        return action, actiond, actionc, log_prob, d_probs, c_means, c_std, state_value, entropy


    


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
        astd = torch.sigmoid(action_std)*(self.upper_bound - self.lower_bound)*0.5 + 1e-8
        adisc = F.softmax(action_discrete, dim=-1)
        return adisc, amean, astd
    
class Value(torch.nn.Module):
    def __init__(self, input_dim, output_dim_discrete, output_dim_continuous, hidden_dim, lower_bound=0, upper_bound=1):
        super(Value, self).__init__()
        self.output_dim_discrete = output_dim_discrete
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine4 = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(p=0.0)


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
        output = self.affine4(x)
        #x = self.dropout(x)
        return output




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