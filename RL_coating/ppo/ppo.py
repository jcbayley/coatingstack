import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from truncated_normal import TruncatedNormalDist

class RolloutBuffer:
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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, discrete_action_dim, hidden_dim=64, lower_bound=0.1, upper_bound=1.0):
        super(ActorCritic, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.vinput = nn.Linear(state_dim, hidden_dim)
        self.vfc1 = nn.Linear(hidden_dim, hidden_dim)
        self.vfc2 = nn.Linear(hidden_dim, hidden_dim)

        self.actor_discrete = nn.Linear(hidden_dim, discrete_action_dim)
        self.actor_continuous_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_continuous_std = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x1 = F.relu(self.input(x))
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        logits_discrete = F.softmax(self.actor_discrete(x1), dim=-1)
        mean_continuous = F.sigmoid(self.actor_continuous_mean(x1))*(self.upper_bound - self.lower_bound) + self.lower_bound
        std_continuous = F.softplus(self.actor_continuous_std(x1))  + 1e-4# Ensure std is positive

        x2 = F.relu(self.vinput(x))
        x2 = F.relu(self.vfc1(x2))
        x2 = F.relu(self.vfc2(x2))
        value = self.critic(x2)

        return logits_discrete, mean_continuous, std_continuous, value

class PPO:
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            discrete_action_dim, 
            hidden_dim=16,
            lr=3e-4, 
            gamma=0.99, 
            clip_ratio=0.2, 
            value_loss_coef=0.5, 
            entropy_coef=0.01,
            lower_bound=0.1,
            upper_bound=1,
            n_updates=20):
        
        self.lr = lr
        self.actor_critic = ActorCritic(state_dim, action_dim, discrete_action_dim, hidden_dim, lower_bound=lower_bound, upper_bound=upper_bound)
        self.old_actor_critic = ActorCritic(state_dim, action_dim, discrete_action_dim, hidden_dim, lower_bound=lower_bound, upper_bound=upper_bound)
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

        self.optimiser = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()

        self.n_updates = n_updates
        self.device=torch.device("cpu")

    def select_action(self, state):
        logits_discrete, mean_continuous, std_continuous, value = self.actor_critic(torch.FloatTensor(state))
        """
        action_discrete = Categorical(logits=logits_discrete).sample().item()
        action_continuous = TruncatedNormalDist(mean_continuous, std_continuous, self.lower_bound, self.upper_bound).rsample().item()
        while not np.isfinite(action_continuous):
            print("INF_ACTION-------------------", mean_continuous, std_continuous)
            action_continuous = TruncatedNormalDist(mean_continuous, std_continuous, self.lower_bound, self.upper_bound).rsample().item()
        """

        dist_discrete = Categorical(logits=logits_discrete)
        dist_continuous = TruncatedNormalDist(mean_continuous, std_continuous, self.lower_bound, self.upper_bound)
    
        action_discrete = dist_discrete.sample()
        action_continuous = dist_continuous.rsample()

        discrete_logprob = dist_discrete.log_prob(action_discrete)
        #print(mean_continuous, continuous_action)
        continuous_logprob = dist_continuous.log_prob(action_continuous)

        log_prob = discrete_logprob + continuous_logprob
        #print(log_prob)
        entropy = dist_discrete.entropy() #+ c_dist.entropy()

        return (action_discrete.item(), action_continuous.item(), mean_continuous, std_continuous, logits_discrete), (log_prob, value, entropy)
    
    def compute_returns(self, rewards, dones):
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        return returns
    
    def evaluate(self, state, discrete_action, continuous_action):
        logits_discrete, mean_continuous, std_continuous, value = self.actor_critic(torch.FloatTensor(state))
        d_dist = Categorical(logits=logits_discrete)
        c_dist = TruncatedNormalDist(mean_continuous, std_continuous, self.lower_bound, self.upper_bound)

        #print(discrete_action, torch.from_numpy(np.array([discrete_action])))
        discrete_logprob = d_dist.log_prob(torch.from_numpy(np.array([discrete_action])))
        #print(mean_continuous, continuous_action)
        try:
            continuous_logprob = c_dist.log_prob(torch.from_numpy(np.array([continuous_action])))
        except:
            print(continuous_action, mean_continuous, std_continuous, self.lower_bound, self.upper_bound)
            sys.exit()
        #print(state)
        #print(discrete_logprob)
        #print(continuous_logprob)
        log_prob = discrete_logprob + continuous_logprob
        #print(log_prob)
        entropy = d_dist.entropy() #+ c_dist.entropy()

        return log_prob, value, entropy
    
    def update(self,):


        returns = self.compute_returns(self.buffer.rewards, self.buffer.dones)

        old_states = torch.squeeze(torch.from_numpy(np.array(self.buffer.states)), dim=0).detach().to(self.device).to(torch.float32)
        old_discrete_actions = torch.from_numpy(np.array(self.buffer.discrete_actions)).detach().to(self.device)
        old_continuous_actions = torch.from_numpy(np.array(self.buffer.continuous_actions)).detach().to(self.device).to(torch.float32)
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(self.device).to(torch.float32)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values), dim=0).detach().to(self.device).to(torch.float32)

        returns = torch.from_numpy(np.array(returns)).to(torch.float32)
        # calculate advantages
        advantages = returns.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.n_updates):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_discrete_actions, old_continuous_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages

            # final loss of clipped objective PPO
            loss1 = -(torch.min(surr1, surr2)).mean()
            loss2 = (0.5 * self.mse_loss(state_values, returns)).mean()  - 0.01 * dist_entropy
            
            loss = (loss1 + loss2).mean()
            # take gradient step
            self.optimiser.zero_grad()
            loss.mean().backward()
            self.optimiser.step()


        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

        self.buffer.clear()

        return loss1.item(), loss2.item()