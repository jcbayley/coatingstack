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

class Environment(gym.Env):

    def __init__(self, max_layers, min_thickness, max_thickness, materials, thickness_options=[0.1,1,10], variable_layers=False):
        self.variable_layers = variable_layers
        self.max_layers = max_layers
        self.thickness_options = thickness_options
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.materials = materials
        self.n_materials = len(materials)
        self.n_thickness = len(self.thickness_options)
        self.n_actions = self.max_layers*self.n_materials + self.max_layers*self.n_thickness

        # state space size is index for each material (onehot encoded) plus thickness of each material
        self.state_space_size = self.max_layers*self.n_materials + self.max_layers

        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_reward = self.compute_reward(self.current_state)

    def reset(self,):
        """reset the state space and length
        """
        self.length = 0
        self.current_state = self.sample_state_space()

        return self.current_state
    
    def print_state(self,):
        for i in range(len(self.current_state)):
            print(self.current_state[i])

    def sample_state_space(self, ):
        """sample from the available state space

        Returns:
            _type_: _description_
        """
        if self.variable_layers:
            n_layers = np.random.randint(2, self.max_layers)
        else:
            n_layers = self.max_layers

        layers = []
        for layer_ind in range(n_layers):
            material_onehot = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(self.n_materials))), num_classes=self.n_materials)
            thickness = np.random.uniform(self.min_thickness, self.max_thickness)
            layer = np.concatenate([[thickness], material_onehot])
            layers.append(layer)

        return torch.from_numpy(np.array(layers))
    
    def get_actions(self, action):
        """get action from integer

        Args:
            action (_type_): _description_
        """
        if action < self.max_layers*self.n_thickness:
            layer_ind = np.floor(action/self.n_thickness).astype(int)
            thickness_ind = action % self.n_thickness
            thickness_change = self.thickness_options[thickness_ind]
            material = None

        if action >= self.max_layers*self.n_thickness:
            layer_ind = np.floor((action - self.max_layers*self.n_thickness)/self.n_materials).astype(int)
            material_ind = action % self.n_materials
            material = self.numpy_onehot(material_ind, num_classes=self.n_materials)
            thickness_change = None

        return layer_ind, material, thickness_change

    
    def numpy_onehot(self, numpy_int, num_classes):
        onehot = torch.nn.functional.one_hot(torch.from_numpy(np.array(numpy_int)), num_classes=num_classes)
        return onehot
        
    def sample_action_space(self, ):
        """sample from the action space

        Returns:
            _type_: _description_
        """

        #action = self.numpy_onehot(np.random.randint(0,self.n_actions), num_classes=self.n_actions)
        action = np.random.randint(0,self.n_actions)
        return action

    def compute_reward(self,state):
        """_summary_

        Args:
            state (_type_): _description_
        """

        reward = 0
        rewards = []
        #print(len(state))
        for i,layer in enumerate(state):
            last_material = self.materials[torch.argmax(state[i-1][1:]).item()]
            current_material = self.materials[torch.argmax(state[i][1:]).item()]
            last_thickness = state[i-1][0]
            current_thickness = state[i][0]
            if current_thickness > self.max_thickness:
                #refract_diff = -10*np.abs(current_thickness - self.max_thickness)
                refract_diff = -10#self.max_thickness
            elif current_thickness < self.min_thickness:
                #refract_diff = -10*np.abs(current_thickness - self.min_thickness)
                refract_diff = -10#self.min_thickness
            else:
                refract_diff = current_material["n"] + np.exp(-(current_thickness - 6)**2/0.5) #+ last_material["n"]*last_thickness

            #print("rdiff", refract_diff)
            rewards.append(refract_diff)
            reward += refract_diff

       
        return reward

    def get_new_state(self, current_state, action):
        """new state is the current action choice

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
 
        #actions = self.get_actions(action)
        #layer = torch.argmax(action[:,:self.max_layers]).item()
        layer = action[0]
        material = action[1]
        thickness_change = action[2]
        #material = action[:,self.max_layers:self.max_layers+self.n_materials]
        #thickness_change = action[:,self.max_layers+self.n_materials:].item()
        if thickness_change is not None:
            current_state[layer][0] += thickness_change    
        if material is not None:
            current_state[layer][1:] = material

        return current_state


    def step(self, action, verbose=False):
        """action[0] - thickness
           action[1:N] - material probability

        Args:
            action (_type_): _description_
        """
        
        actions = self.get_actions(action)
        new_state = self.get_new_state(torch.clone(self.current_state), actions)

        terminated = False
        #print(torch.any((self.current_state[0] + actions[2]) < self.min_thickness))
        if torch.any((new_state[:,0]) < self.min_thickness) or torch.any((new_state[:,0]) > self.max_thickness):
            #print("out of thickness bounds")
            terminated = True
            #pass
            #reward = -1000000
            #new_state = self.current_state

      
        new_reward = self.compute_reward(new_state) 
        reward = new_reward - self.current_reward - 1

        self.length += 1

        if verbose:
            print(actions)
            print(self.current_state)
            print(new_state)

        self.current_state = new_state
        self.current_reward = new_reward
        #print("Reward", reward.item())
        #self.print_state()

        return new_state, reward, terminated, new_reward

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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_dqn')

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

    root_dir = "./test_2_sepact"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_layers = 5
    min_thickness = 1
    max_thickness = 10

    materials = {
        0:{"n": 1},
        1:{"n": 10},
    }

    thickness_options = [-1,-0.1,0.0,0.1,1]

    env = Environment(n_layers, min_thickness, max_thickness, materials, thickness_options=thickness_options)

    # Get the number of state observations
    state = env.reset()
    n_observations = env.state_space_size

    # define number of actions to be made
    #  choose thickness, choose layer, then choose material (onehot)
    n_actions = env.n_actions

    num_games = 4000
    load_checkpoint = False

    agent = Agent(gamma=0.9, epsilon=1.0, alpha=5e-6,
                  input_dims=[n_observations], n_actions=n_actions, mem_size=100000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-5, replace=20,
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
            # Set the episode to end if the reward is within 1 std of the last N
            score += reward #* 1./(runsteps)
            reward_window.append(new_reward)
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
    ax.plot(training_reward_iter[-1])
    ax.plot(training_reward_iter[-10])
    ax.plot(training_reward_iter[-100])

    fig.savefig(os.path.join(root_dir, "training_reward_evolution.png"))

    print("Final state")
    print(final_state)

    num_test = 3
    nsteps = 1000
    output_observations = np.zeros((num_test, *np.shape(final_state)))
    output_scores = np.zeros((num_test, 1))
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
            observation_, reward, done = env.step(action, verbose=False)
        
            #print(f"Reward: {reward}")
            score += reward
            score_evolutions[i, step_ind] = reward
            step_ind += 1
            if step_ind >= nsteps:
                done = True
        print("Nsteps", step_ind)
        
        output_observations[i] = observation_
        output_scores[i] = score

    print(output_observations)
    print(output_scores)

    fig, ax = plt.subplots()
    for i in range(num_test):
        ax.plot(score_evolutions[i])
    
    fig.savefig(os.path.join(root_dir, "rewards.png"))

    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)