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
        self.logprobs_discrete = []
        self.logprobs_continuous = []
        self.rewards = []
        self.state_values = []
        self.dones = []
        self.entropy_discrete = []
        self.entropy_continuous = []
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
        del self.logprobs_discrete[:]
        del self.logprobs_continuous[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]
        del self.entropy_discrete[:]
        del self.entropy_continuous[:]
        del self.returns[:]

    def update(
            self, 
            discrete_action, 
            continuous_action, 
            state, 
            logprob_discrete,
            logprob_continuous, 
            reward, 
            state_value, 
            done,
            entropy_discrete,
            entropy_continuous):
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.states.append(state)
        self.logprobs_discrete.append(logprob_discrete)
        self.logprobs_continuous.append(logprob_continuous)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)
        self.entropy_discrete.append(entropy_discrete)
        self.entropy_continuous.append(entropy_continuous)

    def update_returns(self, returns):
        self.returns.extend(returns)



class PPO(object):

    def __init__(
            self, 
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size, 
            lr_policy=1e-4, 
            lr_value=2e-4, 
            lower_bound=0.1, 
            upper_bound=1.0, 
            n_updates=1, 
            beta=0.1,
            clip_ratio=0.5):

        print("sd", state_dim)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.policy_discrete = DiscretePolicy(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound)
        
        self.policy_continuous = ContinuousPolicy(
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
        
        self.policy_old_discrete = DiscretePolicy(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound)
        
        self.policy_old_continuous = ContinuousPolicy(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound)
        
        self.value_old = Value(
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound)
        
        self.policy_old_discrete.load_state_dict(self.policy_discrete.state_dict())
        self.policy_old_continuous.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        
        self.lr_value = lr_value
        self.lr_policy = lr_policy
        self.optimiser_discrete = torch.optim.Adam(self.policy_discrete.parameters(), lr=self.lr_policy)
        self.optimiser_continuous = torch.optim.Adam(self.policy_continuous.parameters(), lr=self.lr_policy)
        self.optimiser_value = torch.optim.Adam(self.value.parameters(), lr=self.lr_value)

        self.mse_loss = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

        self.beta = beta
        self.clip_ratio = clip_ratio

        self.n_updates = n_updates

    def get_returns(self, rewards):

        temp_r = deque()
        R=0
        for r in rewards[::-1]:
            R = r + 0.99 * R
            temp_r.appendleft(R)
        
        return np.array(temp_r)


    def update(self, update_policy=True, update_value=True):

        R = 0
        policy_loss = []

        eps = 1e-8

        returns = torch.from_numpy(np.array(self.replay_buffer.returns))
        returns = (returns - returns.mean()) / (returns.std() + eps)

        #print(self.replay_buffer.state_values)
        state_vals = torch.cat(self.replay_buffer.state_values).to(torch.float32)
        old_lprobs_discrete = torch.cat(self.replay_buffer.logprobs_discrete).to(torch.float32).detach()
        old_lprobs_continuous = torch.cat(self.replay_buffer.logprobs_continuous).to(torch.float32).detach()
        #advantage = returns.detach() - state_vals.detach()

        #for log_prob, R in zip(self.replay_buffer.logprobs, returns):
        #    policy_loss.append(-log_prob * R)
        #policy_loss = torch.cat(policy_loss).mean()
        #print(torch.cat(self.replay_buffer.logprobs).size(), returns.size())
        #print(self.replay_buffer.continuous_actions)
        actionsc = torch.cat(self.replay_buffer.continuous_actions, dim=0).detach()
        actionsd = torch.cat(self.replay_buffer.discrete_actions, dim=-1).detach()
        #print(actionsd.size(), actionsc.size())
        #torch.autograd.set_detect_anomaly(True)
        for _ in range(self.n_updates):
            # compute probs values and advantages
            states = torch.tensor(self.replay_buffer.states).to(torch.float32)
            action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_probs, c_means, c_std, state_value, entropy_discrete, entropy_continuous = self.select_action(states, actionsc, actionsd)
            advantage = returns.detach() - state_value.detach()

            # compute discrete PPO clipped objective
            ratios_discrete = torch.exp(log_prob_discrete - old_lprobs_discrete)
            #policy_loss_discrete = (-ratios_discrete.squeeze() * advantage.squeeze() - self.beta*entropy_discrete.squeeze()).mean()
            d_surr1 = ratios_discrete.squeeze() * advantage.squeeze()
            d_surr2 = torch.clamp(ratios_discrete, 1-self.clip_ratio, 1+self.clip_ratio) * advantage.squeeze()
            policy_loss_discrete = -(torch.min(d_surr1, d_surr2) + self.beta*entropy_discrete.squeeze()).mean()
 
            # compute continuous PPO clipped objective
            ratios_continuous = torch.exp(log_prob_continuous - old_lprobs_continuous)
            c_surr1 = ratios_continuous.squeeze() * advantage.squeeze()
            c_surr2 = torch.clamp(ratios_continuous, 1-self.clip_ratio, 1+self.clip_ratio) * advantage.squeeze()
            policy_loss_continuous = -(torch.min(c_surr1, c_surr2) + self.beta*entropy_continuous.squeeze()).mean()
            #policy_loss_continuous = -(torch.min(c_surr1, c_surr2)).mean()
            #policy_loss_continuous = -(log_prob_continuous * advantage.squeeze() + self.beta*entropy_continuous.squeeze()).mean()

   
            value_loss = self.mse_loss(returns.to(torch.float32).squeeze(), state_value.squeeze())

            #print(policy_loss_discrete, policy_loss_continuous, value_loss)
            if update_policy:
                self.optimiser_discrete.zero_grad()
                policy_loss_discrete.backward()
                self.optimiser_discrete.step()

            self.optimiser_continuous.zero_grad()
            policy_loss_continuous.backward()
            self.optimiser_continuous.step()

            if update_value:
                self.optimiser_value.zero_grad()
                value_loss.backward()
                self.optimiser_value.step()

        self.policy_old_discrete.load_state_dict(self.policy_discrete.state_dict())
        self.policy_old_continuous.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        self.replay_buffer.clear()

        return policy_loss_discrete.item(), policy_loss_continuous.item(), value_loss.item()
    

    def select_action(self, state, actionc=None, actiond=None):

        if type(state) in [np.array, np.ndarray]:
            state = torch.from_numpy(state).flatten().unsqueeze(0).to(torch.float32)

        d_probs = self.policy_discrete(state)
        c_means, c_std = self.policy_continuous(state)

        state_value = self.value(state)
        #d_onehot = F.one_hot(d_probs, num_classes=policy.output_dim_discrete)
        d = torch.distributions.Categorical(d_probs)

        if actiond is None:
            actiond = d.sample()

        
        c = TruncatedNormalDist(
            c_means, 
            c_std, 
            self.lower_bound, 
            self.upper_bound)
        """
        c = torch.distributions.Normal(
            c_means, 
            c_std
            )
        """
        if actionc is None:
            actionc = c.sample()

        #actionc[actionc == self.lower_bound] += 1e-4
        #actionc[actionc == self.upper_bound] -= 1e-4

        #print(d_probs.size(), actiond.size(), actionc.size(), c_means.size())
        
        log_prob_discrete = d.log_prob(actiond) 
        log_prob_continuous = torch.sum(c.log_prob(actionc), dim=-1)#[:, actiond.detach()]

        # get the continuous action for sampled discrete element
        
        c_action = actionc.detach()[:,actiond.detach()]

        #print(actiond.unsqueeze(0).T.size(), actionc.size())
        action = torch.cat([actiond.detach().unsqueeze(0).T, c_action], dim=-1)[0]

        entropy_discrete = d.entropy() 
        entropy_continuous = torch.sum(c._entropy, dim=-1)

        #policy.saved_log_probs.append(log_prob)

        return action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_probs, c_means, c_std, state_value, entropy_discrete, entropy_continuous


    


class DiscretePolicy(torch.nn.Module):
    def __init__(self, input_dim, output_dim_discrete, output_dim_continuous, hidden_dim, lower_bound=0, upper_bound=1):
        super(DiscretePolicy, self).__init__()
        self.output_dim_discrete = output_dim_discrete
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.0)
        self.output_discrete = torch.nn.Linear(hidden_dim, output_dim_discrete)

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
        #astd = torch.nn.functional.softplus(action_std) + 1e-5
        adisc = F.softmax(action_discrete, dim=-1)
        return adisc
    
class ContinuousPolicy(torch.nn.Module):
    def __init__(self, input_dim, output_dim_discrete, output_dim_continuous, hidden_dim, lower_bound=0, upper_bound=1):
        super(ContinuousPolicy, self).__init__()
        self.output_dim_discrete = output_dim_discrete
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.0)
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
        action_mean = self.output_continuous_mean(x)
        action_std = self.output_continuous_std(x)
        amean = torch.sigmoid(action_mean)*(self.upper_bound - self.lower_bound) + self.lower_bound
        astd = torch.sigmoid(action_std)*(self.upper_bound - self.lower_bound)*0.1 + 1e-6
        return amean, astd
    
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