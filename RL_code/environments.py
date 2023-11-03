import torch
import numpy as np
from coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2
import time

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

    
    def compute_state_value(self, state, material_sub=1, lambda_=1064E-9, f=100, wBeam=0.062, Temp=293):
        """_summary_

        Args:
            state (_type_): _description_
            material_sub (int, optional): Substrate material type 
            
            lambda_ (int, optional): laser wavelength (m)
            f (int, optional): frequency of interest (Hz)
            wBeam (int, optional): laser beam radius on optic(m)
            Temp (int, optional): detector temperature (deg)

        Returns:
            _type_: stuff
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
        current_material = -1
        for i,layer in enumerate(state):
            mat = np.argmax(layer[1:]) + 1
            if current_material == mat:
                return -1e2
            current_material = mat
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
        
        # change units to ASD 
        SbrZ = np.sqrt(SbrZ)
        StoZ= np.sqrt(StoZ)
        SteZ = np.sqrt(SteZ)
        StrZ = np.sqrt(StrZ)
        
        #print("R", rCoat)
        #print("coat", absCoat)
        #print(rho)
        #print(SbrZ)
        
        
        aligoCTN = 2.4E-24 

        """
        if rCoat<0: 
            rCoat = -1
        else:
            rCoat = np.log(rCoat)
        """ 
        
        stat = rCoat - 1e-3*(SbrZ/ aligoCTN)
        
        #print(rCoat)
        #print(np.abs(rCoat))
        #print(np.abs(rCoat), np.mean(SbrZ)*1e38, stat)
        if np.isnan(stat) or np.any(d_opt > self.max_thickness) or np.any(d_opt < self.min_thickness):
            return -1e2
        else:    
            return stat
        """
        if np.any(d_opt > self.max_thickness) or np.any(d_opt < self.min_thickness):
            return -1e5
        else:
            return stat
        """

    def compute_reward(self, new_state, old_state):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_value = self.compute_state_value(new_state) + 5
        
        old_value = self.compute_state_value(old_state) + 5
        reward_diff = new_value - old_value 
        
        
        if reward_diff <= 0:
            reward = -1
        else:
            #if new_value < 0.2:
            #    reward = 0
            #else:
            if new_value == -1e2:
                reward = -1
            else:
                reward = new_value
        """

        if new_value == -1e2:
            reward = -1
        else:
            reward = new_value
        #reward = new_value
        #reward_diff = new_value
        #if reward < 0.2:
        #    reward = 0s
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
            terminated = False
            #pass
            reward = -1e2
            #new_state = self.current_state
        else:
            self.current_state = new_state
            self.current_state_value = reward

        self.length += 1

        if verbose:
            print(actions)
            print(self.current_state)
            print(new_state)


        return new_state, reward, terminated, new_value


if __name__ == "__main__":
    
    max_layers = 5
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
        },
    }
    
    cs = CoatingStack(max_layers, min_thickness, max_thickness, materials, thickness_options=[0.1,1,10], variable_layers=False)

    state = cs.sample_state_space()

    val = cs.compute_state_value(state)

    ### TEST vaild arangements

    def generate_valid_arrangements(materials, num_layers):
        def backtrack(arrangement):
            if len(arrangement) == num_layers:
                valid_arrangements.append(arrangement)
                return
            for material in materials:
                if not arrangement or arrangement[-1] != material:
                    backtrack(arrangement + [material])

        valid_arrangements = []
        backtrack([])
        return valid_arrangements

    # Define materials and number of layers
    materials = ['0','1','2']  # Replace '...' with the rest of your materials
    num_layers = 20

    # Generate valid arrangements
    st = time.time()
    valid_arrangements = generate_valid_arrangements(materials, num_layers)
    print("time: ", time.time() - st)
    print(np.shape(valid_arrangements))
    # Print a sample of valid arrangements (to avoid printing too many)
    for arrangement in valid_arrangements[:10]:
        print(arrangement)
