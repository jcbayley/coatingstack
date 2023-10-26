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

        return np.array(layers)
    
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

    def old_compute_state_value(self,state):
        """_summary_

        Args:
            state (_type_): _description_
        """

        reward = 0
        rewards = []
        #print(len(state))
        for i,layer in enumerate(state):
            if i == 0:
                continue
            last_material = self.materials[np.argmax(state[i-1][1:]) + 1]
            current_material = self.materials[np.argmax(state[i][1:]) + 1]
            last_thickness = state[i-1][0]
            current_thickness = state[i][0]
            if current_thickness > self.max_thickness or last_thickness > self.max_thickness:
                #refract_diff = -10*np.abs(current_thickness - self.max_thickness)
                refract_diff = -100#self.max_thickness
            elif current_thickness < self.min_thickness or last_thickness < self.min_thickness:
                #refract_diff = -10*np.abs(current_thickness - self.min_thickness)
                refract_diff = -100#self.min_thickness
            else:
                #refract_diff = current_material["n"]*np.exp(-(current_thickness - 6)**2/0.5) #+ last_material["n"]*last_thickness
                refract_diff = current_material["n"]/current_thickness - last_material["n"]/last_thickness
            #print("rdiff", refract_diff)
            rewards.append(refract_diff)
            reward += refract_diff

       
        return reward
    
    def compute_state_value(self, state, material_sub=1, lambda_=1, f=1, wBeam=1, Temp=1):


        # Extract substrate properties
        nSub = self.materials[material_sub]['n']
        ySub = self.materials[material_sub]['Y']
        pratSub = self.materials[material_sub]['prat']

        # Initialize vectors of material properties
        # Initialize vectors of material properties
        nLayer = np.zeros(self.max_layers)
        aLayer = np.zeros(self.max_layers)
        material_layers = np.zeros(len(state))
        d_opt = np.zeros(len(state))
        for i,layer in enumerate(state):
            mat = np.argmax(layer[1:]) + 1
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
        stat = np.abs(rCoat) + absCoat + np.mean(SbrZ) + np.mean(StoZ) + np.mean(SteZ) 
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
            reward_diff = -1

        return reward_diff, new_value
    
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
        #print(torch.any((self.current_state[0] + actions[2]) < self.min_thickness))
        if torch.any((self.current_state[:,0] + actions[2]) < self.min_thickness) or torch.any((self.current_state[:,0] + actions[2]) > self.max_thickness):
            #print("out of thickness bounds")
            terminated = True
            #pass
            #reward = -1000000
            #new_state = self.current_state

        #temp_current_state = torch.clone(self.current_state)
        new_state = self.get_new_state(torch.clone(self.current_state), actions)
      
        reward, state_value = self.compute_reward(new_state, self.current_state)

        self.length += 1

        if verbose:
            print(actions)
            print(self.current_state)
            print(new_state)

        self.current_state = new_state
        self.current_state_value = state_value
        #print("Reward", reward.item())
        #self.print_state()

        return new_state, reward, terminated, state_value

class StatePool():

    def __init__(self, environment, n_states=50, n_keep_states = 10):
        self.environment = environment
        self.n_states = n_states
        self.n_keep_states = n_keep_states
        if self.n_states % self.n_keep_states != 0:
            raise Exception(f"keep states must divide into n states")
        self.fraction_keep = int(self.n_states/self.n_keep_states)
        self.current_states = self.get_new_states(self.n_states)

    def order_states(self, ):

        state_values = []
        for i in range(self.n_states):
            temp_stval = self.environment.compute_state_value(self.current_states[i])
            state_values.append((i,temp_stval))

        sorted_state_values = sorted(state_values, key=lambda a: -a[1])

        return np.array(sorted_state_values)

    def evolve_state(self, state):
        action = np.random.randint(self.environment.n_actions)
        actions = self.environment.get_actions(action)
        new_state = self.environment.get_new_state(state, actions)
        return new_state

    def evolve_step(self,):

        sorted_state_values = self.order_states()
        top_state_values = sorted_state_values[:self.n_keep_states]
        top_states = self.current_states[top_state_values[:,0].astype(int)]
        self.current_states = np.tile(top_states, (self.fraction_keep, 1, 1))
        for i in range(self.n_states):
            self.current_states[i] = self.evolve_state(self.current_states[i])

        return top_state_values

    def get_new_states(self, N):
        states = []
        for i in range(N):
            states.append(self.environment.sample_state_space())
        return np.array(states)
    




if __name__ == '__main__':
    #env = gym.make('CartPole-v1')

    root_dir = "./genetic_1"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_layers = 5
    min_thickness = 0.1
    max_thickness = 1

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
        }
    }

    thickness_options = [-1,-0.1,0.0,0.1,1]

    env = Environment(
        n_layers, 
        min_thickness, 
        max_thickness, 
        materials, 
        thickness_options=thickness_options)
    
    num_iterations = 300
    statepool = StatePool(env, n_states=500, n_keep_states = 50)



    filename = os.path.join(root_dir,'coating.png')
    scores = []
    eps_history = []
    n_steps = 0
    final_state = None
    n_mean_calc = 100
    for i in range(num_iterations):
        sort_state_values = statepool.evolve_step()
        score = np.mean(sort_state_values[:,1])
        scores.append(score)
        if i % 10 == 0:
            print('episode: ', i,'score %.1f ' % score)

    sorted_state_values = statepool.order_states()
    top_states = statepool.current_states[sorted_state_values[:10, 0].astype(int)]
    print(top_states)
    print(sorted_state_values[:10])

