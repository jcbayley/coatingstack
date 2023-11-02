import torch
import numpy as np
from coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2

class CoatingStack():

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
        #onehot_action = np.zeros(self.n_actions)
        #onehot_action[action] = 1
        repos = onehot_action.reshape(self.max_layers, self.n_materials, self.n_thickness)
        actionind = np.argmax(repos)
        actions = np.unravel_index(actionind, repos.shape)
        # should give 33 numbers (which layer, which material, which thickness change)

        return actions
    
    def numpy_onehot(self, numpy_int, num_classes):
        onehot = torch.nn.functional.one_hot(torch.from_numpy(np.array(numpy_int)), num_classes=num_classes)
        return onehot
    
    def get_actions(self, action):
        """get the physical actions from the indices of the action

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        actions = self.action_onehot_to_position(action)
    
        layer = actions[0].item()
        #material = np.zeros(self.n_materials)
        #material[actions[1]] = 1
        #material = torch.nn.functional.one_hot(torch.from_numpy(np.array([actions[1]])), num_classes=self.n_materials)
        material = self.numpy_onehot(actions[1], num_classes=self.n_materials)
        thickness_change = self.thickness_options[actions[2].item()]

        return layer, material, thickness_change
    
        
    def sample_action_space(self, ):
        """sample from the action space

        Returns:
            _type_: _description_
        """

        #action = self.numpy_onehot(np.random.randint(0,self.n_actions), num_classes=self.n_actions)
        action = np.random.randint(0,self.n_actions)
        return action

    
    def compute_state_value(self, state, material_sub=1, lambda_=1, f=1, wBeam=1, Temp=1):
        """_summary_

        Args:
            state (_type_): _description_
            material_sub (int, optional): _description_. Defaults to 1.
            lambda_ (int, optional): _description_. Defaults to 1.
            f (int, optional): _description_. Defaults to 1.
            wBeam (int, optional): _description_. Defaults to 1.
            Temp (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """

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
        stat = np.abs(rCoat) #- np.mean(SbrZ)*1e37 +5
        if stat < 0:
            print("STAT LESS THAN ZERO")
        #print(np.abs(rCoat), np.mean(SbrZ)*1e38, stat)
        if np.any(d_opt > self.max_thickness) or np.any(d_opt < self.min_thickness):
            return -1
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

        if reward_diff <= 0:
            reward = -0.1
        else:
            if new_value < 0.2:
                reward = 0
            else:
                reward = new_value
        """
        reward = new_value
        reward_diff = new_value
        if reward < 0.2:
            reward = 0
        """
        return reward_diff, reward, new_value
    
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

        new_state = self.get_new_state(np.copy(self.current_state), actions)
      
        reward_diff, reward, new_value = self.compute_reward(new_state, self.current_state)


        terminated = False
        #print(torch.any((self.current_state[0] + actions[2]) < self.min_thickness))
        if np.any((self.current_state[:,0] + actions[2]) < self.min_thickness) or np.any((self.current_state[:,0] + actions[2]) > self.max_thickness):
            #print("out of thickness bounds")
            terminated = True
            #pass
            reward = -1
            #new_state = self.current_state

        self.length += 1

        if verbose:
            print(actions)
            print(self.current_state)
            print(new_state)

        self.current_state = new_state
        self.current_state_value = reward
        #print("Reward", reward.item())
        #self.print_state()

        return new_state, reward, terminated, new_value
