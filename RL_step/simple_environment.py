import torch
import numpy as np
#from coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2, merit_function
import time

class CoatingStack():

    def __init__(
            self, 
            max_layers, 
            min_thickness, 
            max_thickness, 
            materials, 
            air_material=None,
            thickness_options=[0.1,1,10], 
            variable_layers=False):
        """_summary_

        Args:
            max_layers (_type_): _description_
            min_thickness (_type_): _description_
            max_thickness (_type_): _description_
            materials (_type_): _description_
            air_material (_type_): _description_
            thickness_options (list, optional): _description_. Defaults to [0.1,1,10].
            variable_layers (bool, optional): _description_. Defaults to False.
        """
        self.variable_layers = variable_layers
        self.max_layers = max_layers
        self.thickness_options = thickness_options
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.materials = materials
        self.air_material = air_material
        self.n_materials = len(materials)
        self.n_thickness = len(self.thickness_options)
        self.n_actions = self.max_layers*self.n_materials*self.n_thickness

        # state space size is index for each material (onehot encoded) plus thickness of each material
        self.state_space_size = self.max_layers*self.n_materials + self.max_layers

        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = 0
        self.previous_material = -1

    def reset(self,):
        """reset the state space and length
        """
        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = 0

        return self.current_state
    
    def print_state(self,):
        for i in range(len(self.current_state)):
            print(self.current_state[i])

    def sample_state_space(self, ):
        """return air with a thickness of 1

        Returns:
            _type_: _description_
        """
        layers = np.zeros((self.max_layers, self.n_materials + 1))
        layers[:,1] = 1
        layers[:,0] = self.min_thickness
        return layers

    def sample_action_space(self, ):
        """sample from the available state space

        Returns:
            _type_: _description_
        """

        new_layer_material = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(self.n_materials))), num_classes=self.n_materials)
        new_layer_thickness = torch.random.uniform(self.min_thickness, self.max_thickness)
        new_layer = torch.cat([new_layer_thickness, new_layer_material])

        return new_layer

    
        
    def sample_action_space(self, ):
        """sample from the action space

        Returns:
            _type_: _description_
        """

        #action = self.numpy_onehot(np.random.randint(0,self.n_actions), num_classes=self.n_actions)
        material = torch.random.randint(0,self.n_materials)
        thickness = torch.random.uniform(self.min_thickness, self.max_thickness)
        return thickness, material

    
    def compute_state_value(
            self, 
            state, 
            material_sub=1, 
            laser_wavelength=1064E-9, 
            frequency=100, 
            wBeam=0.062, 
            Temp=293):
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

        m_scaled = 0
        last_material = None
        for i, layer in enumerate(state):
            material_ind = np.argmax(layer[1:])
            material = self.materials[material_ind]
            m_scaled += -np.abs((0.25 - state[i][0]))
            if last_material is not None:
                m_scaled += material["n"]/last_material["n"]
            last_material = material
    
        return m_scaled

    def compute_reward(self, new_state, max_value=0.0):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_value = self.compute_state_value(new_state)
        #old_value = self.compute_state_value(old_state) + 5
        reward_diff = new_value - max_value

        if reward_diff > 0:
            #print(reward_diff)
            reward = reward_diff
        else:
            reward = reward_diff
 
        return reward_diff, reward, new_value 
    
    def update_state(self, current_state, action):
        """new state is the current action choice

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        material = torch.nn.functional.one_hot(action[1], num_classes=self.n_materials)[0]
        thickness = action[0]
        new_layer = torch.cat([thickness, material])
        current_state[self.current_index] = new_layer

        return current_state


    def step(self, action, max_state, verbose=False):
        """action[0] - thickness
           action[1:N] - material probability

        Args:
            action (_type_): _description_
        """
        
        thickness_change = action[0]
        material_change = action[1]
            

        new_state = self.update_state(np.copy(self.current_state), action)
      
        reward = 0
        reward_diff, reward, new_value = self.compute_reward(new_state, max_state)


        terminated = False
        #print(torch.any((self.current_state[0] + actions[2]) < self.min_thickness))
        if thickness_change <=0:
            terminated=True
            reward = -0.1
        elif self.current_index >= self.max_layers-1 or action[1][0] == 0:
         #print("out of thickness bounds")
            terminated = True
            #reward_diff, reward, new_value = self.compute_reward(new_state, max_state)
        #elif action[1][0] == self.previous_material:
        #    terminated = True
        #    reward = -0.01
        else:
            self.current_state = new_state
            #self.current_state_value = reward

        self.previous_material = action[1][0]
        #print(new_value)

        self.length += 1
        self.current_index += 1

        #print("cind:", self.current_index)
        #print(new_state)


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

