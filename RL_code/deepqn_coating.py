import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt
from deepqn_cart import plotLearning
import copy 
from simple_environment import CoatingStack

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DuelingLinearDeepQNetwork(nn.Module):
    def __init__(self, ALPHA, n_actions, name, input_dims, chkpt_dir='tmp/dqn'):
        super(DuelingLinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_dqn.pt')

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, gamma, epsilon, alpha, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DuelingLinearDeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_eval', chkpt_dir=chkpt_dir)
        self.q_next = DuelingLinearDeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_next', chkpt_dir=chkpt_dir)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis,:]
            state = torch.tensor(observation).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state.flatten(start_dim=1).to(torch.float32))
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                         if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        # using torch.Tensor seems to reset datatype to float
        # using torch.tensor preserves source data type
        state = torch.tensor(state).to(self.q_eval.device)
        new_state = torch.tensor(new_state).to(self.q_eval.device)
        action = torch.tensor(action).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)

        V_s, A_s = self.q_eval.forward(state.flatten(start_dim=1).to(torch.float32))
        V_s_, A_s_ = self.q_next.forward(new_state.flatten(start_dim=1).to(torch.float32))

        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1,
                                              action.unsqueeze(-1)).squeeze(-1)

        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_target = rewards + self.gamma*torch.max(q_next, dim=1)[0].detach()
        q_target[dones.bool()] = 0.0

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


if __name__ == '__main__':
    #env = gym.make('CartPole-v1')

    root_dir = "./ddqn_1_allopt"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_layers = 5
    min_thickness = 0.01
    max_thickness = 2

    materials = {
        1: {
            'name': 'SiO2',
            'n': 1.44,
            'a': 0,
            'alpha': 0.51e-6,
            'beta': 8e-6,
            'kappa': 1.38,
            'C': 1.64e6,
            'Y': 72e9,
            'prat': 0.17,
            'phiM': 4.6e-5
        },
        2: {
            'name': 'ta2o5',
            'n': 2.07,
            'a': 2,
            'alpha': 3.6e-6,
            'beta': 14e-6,
            'kappa': 33,
            'C': 2.1e6,
            'Y': 140e9,
            'prat': 0.23,
            'phiM': 2.44e-4
        },
    }

    thickness_options = [-0.1,-0.01,0.0,0.01,0.1]

    env = CoatingStack(
        n_layers, 
        min_thickness, 
        max_thickness, 
        materials, 
        thickness_options=thickness_options)
    

    # Get the number of state observations
    state = env.reset()
    n_observations = env.state_space_size

    # define number of actions to be made
    #  choose thickness, choose layer, then choose material (onehot)
    n_actions = env.n_actions

    num_games = 4000
    load_checkpoint = False

    agent = Agent(gamma=0.9, 
                  epsilon=1.0, 
                  alpha=1e-5,
                  input_dims=[n_observations], 
                  n_actions=n_actions, 
                  mem_size=100000, 
                  eps_min=0.01,
                  batch_size=128, 
                  eps_dec=1e-4,   # decay rate of random steps
                  replace=100,    # how often to replace target with eval network
                  chkpt_dir=root_dir)

    if load_checkpoint:
        agent.load_models()

    filename = os.path.join(root_dir,'coating.png')
    scores = []
    eps_history = []
    training_reward_iter = []
    n_steps = 0
    final_state = None
    n_mean_calc = 100
    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0
        reward_window = []
        runsteps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, new_reward = env.step(action)
            n_steps += 1
            runsteps += 1
            reward_window.append(new_reward)
            # Set the episode to end if the reward is within 1 std of the last N
            score += reward #* 1./(runsteps)
            if runsteps > 1000:
                done=True
            """
            if runsteps < n_mean_calc:
                reward_window.append(reward)
            else:
                reward_window.pop(0)
                reward_window.append(reward)

            if reward - np.mean(reward_window) < np.std(reward_window):
                done = True
            """
            agent.store_transition(observation.flatten(), action,
                                    reward, observation_.flatten(), int(done))
            agent.learn()

            observation = observation_

            final_state = observation

        training_reward_iter.append(reward_window)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        if i % 10 == 0:
            print('episode: ', i,'score %.1f ' % score,
                ' average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        if i > 0 and i % 100 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)

    fig, ax = plt.subplots()
    for ind in [10, int(0.5*len(training_reward_iter)), len(training_reward_iter)-10]:
        ax.plot(training_reward_iter[ind][:100], label=f"episode: {ind}")
    ax.legend()

    fig.savefig(os.path.join(root_dir, "training_reward_evolution_zoomstart.png"))

    fig, ax = plt.subplots()
    for ind in [10, int(0.5*len(training_reward_iter)), len(training_reward_iter)-10]:
        ax.plot(training_reward_iter[ind], label=f"episode: {ind}")
    ax.legend()

    fig.savefig(os.path.join(root_dir, "training_reward_evolution.png"))

    print("Final state")
    print(final_state)

    num_test = 3
    nsteps = 5000
    output_observations = np.zeros((num_test, *np.shape(final_state)))
    output_scores = np.zeros((num_test, 1))
    output_values = np.zeros((num_test, 1))
    score_evolutions = np.zeros((num_test, nsteps))
    for i in range(num_test):

        observation = env.reset()
        print(f"Obs: {i}", observation)
        done = False
        step_ind = 0
        score = 0
        while not done:
            #print("step")
            action = agent.choose_action(observation)
            observation_, reward, done, new_state_value = env.step(action, verbose=False)
        
            #print(f"Reward: {reward}")
            score += reward
            score_evolutions[i, step_ind] = new_state_value
            step_ind += 1
            if step_ind >= nsteps:
                done = True
        print("Nsteps", step_ind)
        
        output_observations[i] = observation_
        output_scores[i] = score
        output_values[i] = new_state_value

    print(output_observations)
    print(output_scores)
    print(output_values)

    fig, ax = plt.subplots()
    for i in range(num_test):
        ax.plot(score_evolutions[i])
    
    fig.savefig(os.path.join(root_dir, "rewards.png"))

    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)