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
from coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2
from torch.distributions import Categorical

class Environment():

    def __init__(self, max_layers, min_thickness, max_thickness, materials, thickness_options=[0.1,1,10], variable_layers=False):
        self.variable_layers = variable_layers
        self.max_layers = max_layers
        self.thickness_options = thickness_options
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.materials = materials
        self.n_materials = len(materials)
        self.n_thickness = len(self.thickness_options)
        self.n_actions = self.max_layers*self.n_materials*self.n_thickness

        # state space size is index for each material (onehot encoded) plus thickness of each material
        self.state_space_size = self.max_layers*self.n_materials + self.max_layers

        self.length = 0
        self.current_state = self.sample_state_space()

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
    
    def action_onehot_to_position(self, action):
        """change from integer action to a list of 3 indices according the the 3 actions

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        onehot_action = torch.nn.functional.one_hot(torch.from_numpy(np.array([action])), num_classes=self.n_actions)
        repos = onehot_action.reshape(self.max_layers, self.n_materials, self.n_thickness)
        actionind = np.argmax(repos)
        actions = np.unravel_index(actionind, repos.shape)
        # should give 33 numbers (which layer, which material, which thickness change)

        return actions
    
    def get_actions(self, action):
        """get the physical actions from the indices of the action

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        actions = self.action_onehot_to_position(action)
    
        layer = actions[0].item()
        material = self.numpy_onehot(actions[1], num_classes=self.n_materials)
        thickness_change = self.thickness_options[actions[2].item()]

        return layer, material, thickness_change
    
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

    
    def compute_state_value(self, state, material_sub=1, lambda_=1, f=1, wBeam=1, Temp=1):


        # Extract substrate properties
        nSub = self.materials[material_sub]['n']
        ySub = self.materials[material_sub]['Y']
        pratSub = self.materials[material_sub]['prat']

        # Initialize vectors of material properties
        nLayer = np.zeros(self.max_layers)
        aLayer = np.zeros(self.max_layers)
        material_layers = np.zeros(len(state))
        d_opt = np.zeros(len(state))
        for i,layer in enumerate(state):
            mat = np.argmax(layer[1:]).item() + 1
            material_layers[i] = mat
            d_opt[i] = layer[0]
            nLayer[i] = self.materials[mat]['n']
            aLayer[i] = self.materials[mat]['a']


        # Compute reflectivities
        rCoat, dcdp, rbar, r = getCoatRefl2(1, nSub, nLayer, d_opt)

        # Compute absorption
        absCoat, absLayer, powerLayer, rho = getCoatAbsorption(lambda_, d_opt, aLayer, nLayer, rbar, r)

        # Compute brownian and thermo-optic noises
        SbrZ, StoZ, SteZ, StrZ, brLayer = getCoatNoise2(f, lambda_, wBeam, Temp, self.materials, material_sub, material_layers, d_opt, dcdp)

        #print("R", rCoat)
        #print("coat", absCoat)
        #print(rho)
        #print(SbrZ)
        #sys.exit()
        stat = np.abs(rCoat) - np.mean(SbrZ)*1e38
        #print(np.abs(rCoat), np.mean(SbrZ)*1e38, stat)
        if np.any(d_opt > self.max_thickness) or np.any(d_opt < self.min_thickness):
            return -50
        else:
            return stat
    
    def compute_reward(self, new_state, old_state):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_value = self.compute_state_value(new_state) 
        old_value = self.compute_state_value(old_state)
        reward_diff = new_value - old_value 
        if reward_diff < 0:
            reward = -1
        else:
            reward = new_value

        return reward_diff, reward
    
    def get_new_state(self, current_states, actions):
        """new state is the current action choice

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
 
        #actions = self.get_actions(action)
        #layer = torch.argmax(action[:,:self.max_layers]).item()
        layer = actions[0]
        material = actions[1]
        thickness_change = actions[2]
        #material = action[:,self.max_layers:self.max_layers+self.n_materials]
        #thickness_change = action[:,self.max_layers+self.n_materials:].item()
        current_states[layer][0] += thickness_change
        current_states[layer][1:] = material

        return current_states


    def step(self, action, verbose=False):
        """action[0] - thickness
           action[1:N] - material probability

        Args:
            action (_type_): _description_
        """
        
        actions = self.get_actions(action)

        terminated = False

        new_state = self.get_new_state(torch.clone(self.current_state), actions)
      
        rewarddiff, reward = self.compute_reward(new_state, self.current_state)

        #print(torch.any((self.current_state[0] + actions[2]) < self.min_thickness))
        if torch.any((self.current_state[:,0] + actions[2]) < self.min_thickness) or torch.any((self.current_state[:,0] + actions[2]) > self.max_thickness):
            #print("out of thickness bounds")
            terminated = True
            reward = -100
            #pass
            #reward = -1000000
            new_state = self.current_state
        else:
            self.current_state = new_state
            self.current_state_value = reward

        #temp_current_state = torch.clone(self.current_state)
        
        self.length += 1

        if verbose:
            print(actions)
            print(self.current_state)
            print(new_state)


        return new_state, reward, terminated, reward


class Policy(nn.Module):
    def __init__(self, state_space_size, n_actions):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(state_space_size, 1024)
        self.layer2 = nn.Linear(1024,1024)
        self.layer3 = nn.Linear(1024,512)
        self.layer4 = nn.Linear(512,128)
        self.layer5 = nn.Linear(128, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return F.softmax(self.layer5(x), dim=1)

    def select_action(self, state):
        state = state.float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

def run_episode(env, policy, render=False):

    observation = env.reset()
    totalreward = 0

    observations = []
    actions = []
    rewards = []
    probs = []

    done = False

    #while not done:
    for _ in range(1000):
        observations.append(observation)

        action = policy.select_action(observation.flatten())
        observation, reward, done, info = env.step(action)
        if done:
            break

        totalreward += reward
        policy.rewards.append(reward)
        actions.append(action)

    #for i, obs in enumerate(observations[::10]):
    #    print(obs)
    #    print(policy.rewards[::10][i])
    
    value = env.compute_state_value(observation)

    return totalreward, np.array(rewards), np.array(observations), np.array(actions), np.array(probs), value

def finish_episode(policy, optimiser, gamma):
    R = 0
    policy_loss = []
    # compute the discounted reward over whole episode
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    # compute advantage
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    # updates to model    
    optimiser.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimiser.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def train(env, policy, optimiser, MAX_EPISODES=1000, gamma=0.99,seed=None, evaluate=False):

    # initialize and policy
    episode_rewards = []

    # train until MAX_EPISODES
    for i in range(MAX_EPISODES):

        # run a single episode
        total_reward, rewards, observations, actions, probs, value = run_episode(env, policy)

        # keep track of episode rewards
        episode_rewards.append(total_reward)

        # update policy
        if len(policy.rewards) > 2:
            finish_episode(policy, optimiser, gamma=gamma)

        print("EP: " + str(i) + " Score: " + str(total_reward) + " " + "Value: " + str(value))


    return episode_rewards, policy

def test_policy(policy, num_iterations):

    total_rewards = []
    total_observations = []
    total_values = []
    for _ in range(num_iterations):
        totalreward, rewards, observations, actions, probs, value = run_episode(env, policy)
        total_rewards.append(totalreward)
        total_values.append(value)
        total_observations.append(observations[-1])

    return total_observations, total_rewards, total_values

if __name__ == '__main__':
    #env = gym.make('CartPole-v1')

    root_dir = "./ppo_real_2"
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

    thickness_options = [-0.1,-0.01,-0.001,0.0,0.001,0.01,0.1]

    env = Environment(
        n_layers, 
        min_thickness, 
        max_thickness, 
        materials, 
        thickness_options=thickness_options)
    
    theta = np.random.rand(4)
    alpha = 0.002
    gamma = 0.7
    policy = Policy(env.state_space_size, env.n_actions)
    optimiser = optim.Adam(policy.parameters(), lr=1e-2)
    
    num_iterations = 100

    episode_rewards, policy = train(env, policy, optimiser, gamma=gamma, MAX_EPISODES=num_iterations, seed=None, evaluate=False)

    fig, ax = plt.subplots()
    ax.plot(episode_rewards)
    fig.savefig(os.path.join(root_dir, "episode_rewards.png"))
    observations, rewards, values = test_policy(policy, 2)

    print(observations)
    print(rewards)
    print(values)

