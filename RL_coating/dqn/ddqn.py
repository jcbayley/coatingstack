import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, discrete_dim, continuous_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.input = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.discrete = nn.Linear(hidden_dim + discrete_dim*continuous_dim, discrete_dim)
        self.continuous = nn.Linear(hidden_dim, discrete_dim*continuous_dim)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        xc = self.continuous(x)
        catx = torch.cat([x, xc], dim=-1)
        xd = self.discrete(catx)
        return xd, xc

class DDQN:
    def __init__(self, state_dim, discrete_dim, thickness_options, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, target_update_freq=100, hidden_dim=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discrete_dim = discrete_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.max_replay_size = 2000
        self.thickness_options = thickness_options
        self.continuous_dim = len(self.thickness_options)

        # Initialize Q-networks
        self.q_network = QNetwork(state_dim, discrete_dim, self.continuous_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, discrete_dim, self.continuous_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize replay buffer
        self.replay_buffer = []
        self.replay_index = 0
        self.first_pass = True

        # Initialize step counter
        self.steps = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            qloc, t_value= random.randint(0, self.discrete_dim - 1), np.random.uniform(0.1,1)
            #print("rn", qloc, t_value)
            return qloc, t_value
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values, t_value = self.q_network(state)
                t_value = t_value.reshape((self.discrete_dim, self.continuous_dim))
                qloc = torch.argmax(q_values).item()
                tloc = torch.argmax(t_value[qloc]).item()
                #print("dn", qloc, tloc)
                return qloc, tloc

    def store_transition(self, transition):
        if self.first_pass:
            self.replay_buffer.append(transition)
        else:
            self.replay_buffer[self.replay_index] = transition

        if self.replay_index >= self.max_replay_size:
            self.replay_index = 0
            if self.first_pass:
                self.first_pass = False
        else:
            if self.first_pass:
                self.replay_index +=1
            else:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*self.replay_buffer)
                self.replay_index = np.argmin(reward_batch)

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample mini-batch from replay buffer
        transitions = random.sample(self.replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

        # Compute Q-values for current states and next states
        q_values, t_values = self.q_network(state_batch)
        next_q_values, next_t_values = self.target_network(next_state_batch)
        next_q_values_max = torch.max(next_q_values, dim=1)[0]
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values_max

        # Gather Q-values for actions taken
        #print(q_values.size(), action_batch.size())
        q_values_action = q_values.gather(1, action_batch[:,0].unsqueeze(1)).squeeze()

        # Compute loss and update Q-network
        loss = torch.mean((q_values_action - target_q_values) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.steps += 1

        return loss.item()