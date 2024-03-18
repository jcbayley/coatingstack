from collections import namedtuple, deque
import random
import torch
import numpy as np


# setup a simple q network 
class QNetwork(torch.nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size):
        super(QNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)

# Define the experience replay buffer
# the replay buffer stores the training data for the network
# it consists of:
# the current state
# the action taken from this state
# the reward that was gained from taking that action from the state
# the state that we transitioned to by performing that action 
# whether the state was a terminal one or not
    
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, buffer_size):
        # this has a buffer of the about quantities of some desired length (usually restricted by memory)
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Deep Q-Learning agent class
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01, buffer_size=10000, batch_size=64, hidden_size=128, update_frequency=100):
        # environment parameters
        self.state_size = state_size
        self.action_size = action_size
        # parameters associated with the action choice probability
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # discount factor
        self.gamma = gamma
        self.update_frequency = update_frequency

        # Q-networks 
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size), None
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return int(torch.argmax(q_values)), None

    def train(self):
        # Sample a batch from experience replay
        batch = self.replay_buffer.sample_batch(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)


        states = torch.FloatTensor(states).to(torch.float32)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).to(torch.float32)
        next_states = torch.FloatTensor(next_states).to(torch.float32)
        dones = torch.FloatTensor(dones)

        # Q-values for current states
        q_values = self.q_network(states)

        # Q-values for next states using target Q-network
        next_q_values = self.target_q_network(next_states).detach()

        # Compute target Q-values
        # this takes the maxmimum over the 
        target_q_values = rewards + self.gamma * (1 - dones) * torch.max(next_q_values, dim=1)[0]

        # Compute loss
        # this is the difference between the target qvalues and those from one (or more) step in the past
        loss = torch.nn.MSELoss()(q_values.gather(1, actions.view(-1, 1)), target_q_values.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        # Update target Q-network periodically
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        # Update target Q-network with the weights of the current Q-network
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        # Add experience to replay buffer
        self.replay_buffer.add_experience(Experience(state, action, reward, next_state, done))

