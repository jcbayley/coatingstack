import torch
import random
import numpy as np
from collections import deque
from truncated_normal import TruncatedNormalDist

class HybridPolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim, hidden_size, min_thickness, max_thickness):
        super(HybridPolicyNetwork, self).__init__()

        self.fcin = torch.nn.Linear(state_dim, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.discrete_head = torch.nn.Linear(hidden_size, discrete_action_dim)
        if continuous_action_dim != 0:
            self.continuous_head_mean = torch.nn.Linear(hidden_size, continuous_action_dim)
            self.continuous_head_std = torch.nn.Linear(hidden_size, continuous_action_dim)
        self.min_thickness=min_thickness
        self.max_thickness=max_thickness
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        self.lsoft = torch.nn.Softmax()

    def forward(self, state):
        l1 = self.fcin(state)
        l1 = self.fc1(l1)
        l1 = self.fc2(l1)
        discrete_logits = self.lsoft(self.discrete_head(l1))
        if self.continuous_action_dim != 0:
            continuous_mean = torch.nn.functional.sigmoid(self.continuous_head_mean(l1))*(self.max_thickness - self.min_thickness) + self.min_thickness
            continuous_std = torch.nn.functional.sigmoid(self.continuous_head_std(l1))*10
        else:
            continuous_mean = 0
            continuous_std = 0
        return discrete_logits, continuous_mean, continuous_std
    
    def sample_action(self, d_logits, c_mean, c_std):
        discrete_dist = torch.distributions.categorical.Categorical(probs=d_logits)
        discrete_action = torch.nn.functional.gumbel_softmax(torch.log(d_logits), hard=True)
        #discrete_action = torch.multinomial(d_logits, num_samples=1)
        discrete_action = discrete_dist.sample()
        discrete_action = torch.nn.functional.one_hot(discrete_action.to(torch.long), num_classes=self.discrete_action_dim)

        if self.continuous_action_dim:
            continuous_dist = TruncatedNormalDist(
                c_mean, 
                c_std + 1e-8, 
                self.min_thickness, 
                self.max_thickness)
            continuous_action = continuous_dist.sample()
        else:
            continuous_action = 0

        return discrete_action, continuous_action
    
    def log_prob(self, d_logits, c_mean, c_std, discrete_val, continuous_val):
        """_summary_

        Args:
            d_logits (_type_): _description_
            c_mean (_type_): _description_
            c_std (_type_): _description_
            discrete_val (_type_): onehot tensor
            continuous_val (_type_): _description_

        Returns:
            _type_: _description_
        """
        discrete_dist = torch.distributions.categorical.Categorical(probs=d_logits)
        continuous_dist = TruncatedNormalDist(
            c_mean, 
            c_std + 1e-8, 
            self.min_thickness, 
            self.max_thickness,
            validate_args=False)
        
        discrete_val = torch.argmax(discrete_val)
        discrete_log_prob = discrete_dist.log_prob(discrete_val)
        continuous_log_prob = continuous_dist.log_prob(continuous_val)

        return discrete_log_prob + continuous_log_prob
    
class ContinuousPolicyNetwork(torch.nn.Module):

    def __init__(
            self, 
            state_dim,
            continuous_action_dim, 
            hidden_size, 
            lower_bound,
            upper_bound):
        super(ContinuousPolicyNetwork, self).__init__()

        self.fcin = torch.nn.Linear(state_dim, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)

        self.continuous_head_mean = torch.nn.Linear(hidden_size, continuous_action_dim)
        self.continuous_head_std = torch.nn.Linear(hidden_size, continuous_action_dim)
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.continuous_action_dim = continuous_action_dim
        self.lsoft = torch.nn.Softmax()

    def forward(self, state):
        l1 = self.fcin(state)
        l1 = self.fc1(l1)
        l1 = self.fc2(l1)
        continuous_mean = torch.nn.functional.sigmoid(self.continuous_head_mean(l1))*(self.upper_bound - self.lower_bound) + self.lower_bound
        continuous_std = torch.nn.functional.sigmoid(self.continuous_head_std(l1))
 
        return continuous_mean, continuous_std
    
    
    def sample_action(self, c_mean, c_std):

        continuous_dist = TruncatedNormalDist(
            c_mean, 
            c_std + 1e-8, 
            self.lower_bound, 
            self.upper_bound
            )
        
        continuous_action = continuous_dist.sample()
        return continuous_action
    
    def log_prob(self, c_mean, c_std, c_val):

        continuous_dist = TruncatedNormalDist(
            c_mean, 
            c_std + 1e-8, 
            self.lower_bound, 
            self.upper_bound,
            validate_args=False)
        
        continuous_log_prob = continuous_dist.log_prob(c_val)

        return continuous_log_prob
    

class DiscretePolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, discrete_action_dim, hidden_size):
        super(DiscretePolicyNetwork, self).__init__()

        self.fcin = torch.nn.Linear(state_dim, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.discrete_head = torch.nn.Linear(hidden_size, discrete_action_dim)
        self.discrete_action_dim = discrete_action_dim
        self.lsoft = torch.nn.Softmax()

    def forward(self, state):
        l1 = self.fcin(state)
        l1 = self.fc1(l1)
        l1 = self.fc2(l1)
        discrete_logits = self.lsoft(self.discrete_head(l1))

        return discrete_logits
    
    def sample_action(self, d_logits, c_mean, c_std):
        discrete_dist = torch.distributions.categorical.Categorical(probs=d_logits)
        discrete_action = torch.nn.functional.gumbel_softmax(torch.log(d_logits), hard=True)
        #discrete_action = torch.multinomial(d_logits, num_samples=1)
        discrete_action = discrete_dist.sample()
        discrete_action = torch.nn.functional.one_hot(discrete_action.to(torch.long), num_classes=self.discrete_action_dim)

        if self.continuous_action_dim:
            continuous_dist = TruncatedNormalDist(
                c_mean, 
                c_std + 1e-8, 
                self.min_thickness, 
                self.max_thickness)
            continuous_action = continuous_dist.sample()
        else:
            continuous_action = 0

        return discrete_action, continuous_action
    
    def log_prob(self, d_logits, c_mean, c_std, discrete_val, continuous_val):
        """_summary_

        Args:
            d_logits (_type_): _description_
            c_mean (_type_): _description_
            c_std (_type_): _description_
            discrete_val (_type_): onehot tensor
            continuous_val (_type_): _description_

        Returns:
            _type_: _description_
        """
        discrete_dist = torch.distributions.categorical.Categorical(probs=d_logits)
        continuous_dist = TruncatedNormalDist(
            c_mean, 
            c_std + 1e-8, 
            self.min_thickness, 
            self.max_thickness,
            validate_args=False)
        
        discrete_val = torch.argmax(discrete_val)
        discrete_log_prob = discrete_dist.log_prob(discrete_val)
        continuous_log_prob = continuous_dist.log_prob(continuous_val)

        return discrete_log_prob + continuous_log_prob
    


class ReplayBufferDeque:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.buffer = deque(sorted(self.buffer, lambda key: key[2], maxlen=self.capacity))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = np.array(actions)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            min_reward_index = min(range(len(self.buffer)), key=lambda i: self.buffer[i][2])
            if experience[2] > self.buffer[min_reward_index][2]:  # Replace if new experience has higher reward
                self.buffer[min_reward_index] = experience

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = np.array(actions)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )


    def __len__(self):
        return len(self.buffer)
    

    
class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fcin = torch.nn.Linear(state_dim + action_dim, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fcout = torch.nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = torch.nn.functional.relu(self.fcin(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fcout(x)
        return x

class QNetworkNoAction(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(QNetworkNoAction, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fcin = torch.nn.Linear(state_dim, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fcout = torch.nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fcin(state))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fcout(x)
        return x
    