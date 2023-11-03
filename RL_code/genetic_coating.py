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
from environments import CoatingStack
import plotting

class StatePool():

    def __init__(self, environment, n_states=50, states_fraction_keep = 0.1, fraction_random_add=0.0):
        self.environment = environment
        self.n_states = n_states
        self.states_fraction_keep = states_fraction_keep
        self.fraction_random_add = fraction_random_add
        #if self.n_states % self.n_keep_states != 0:
        #    raise Exception(f"keep states must divide into n states")
  
        self.current_states = self.get_new_states(self.n_states)

    def order_states(self, ):
        """set the order of states based on their value

        Returns:
            array: array of sorted state values
        """
        state_values = []
        for i in range(self.n_states):
            temp_stval = self.environment.compute_state_value(self.current_states[i])
            state_values.append((i,temp_stval))

        sorted_state_values = sorted(state_values, key=lambda a: -a[1])

        return np.array(sorted_state_values)
    
    def check_state_possible(self, state):
        """make sure that adjacent layers cannot have the same material

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        current_material = -1
        possible = True
        for layer in state:
            new_material = np.argmax(layer[1:])
            if new_material == current_material:
                possible = False
                break
            current_material = new_material

        return possible

    def evolve_state(self, state):
        """Evolve a state, pick a random action and apply this to a state

        Args:
            state (int): action choice

        Returns:
            array: new state to pick
        """
        
        action = np.random.randint(self.environment.n_actions)
        actions = self.environment.get_actions(action)
        new_state = self.environment.get_new_state(state, actions)
        #done = self.check_state_possible(new_state)
        return new_state
    
    def state_crossover(self, states):
        """swap some attributes of the given states

        Args:
            states (array): set of states 

        Returns:
            array: set of states
        """
        n_states, n_layers, n_features = np.shape(states)
        # creates indicies for each of the example states and shuffles them
        data_inds = np.arange(n_states)
        num_swaps = 3
        nswitch = int(n_states/2)
        for i in range(num_swaps):
            np.random.shuffle(data_inds)
            # define that half of states will switch a layer with another state
            ninds_switch_from = data_inds[:nswitch]
            ninds_switch_to = data_inds[nswitch:]
            layers = np.random.randint(n_layers, size=nswitch)
            states[ninds_switch_from, layers] = states[ninds_switch_to, layers]

        return states

    def evolve_step(self,):
        """Evolve one step, by sorting states, taking top n% duplicating them and applying crossover

        Returns:
            _type_: _description_
        """
        sorted_state_values = self.order_states()
        n_keep_states = int(self.states_fraction_keep * self.n_states)
        top_state_values = sorted_state_values[:n_keep_states]
        top_states = self.current_states[top_state_values[:,0].astype(int)]
        self.current_states = np.tile(top_states, (np.ceil(self.n_states/n_keep_states).astype(int), 1, 1))[:self.n_states]
        self.current_states = self.state_crossover(self.current_states)
        
        if self.fraction_random_add != 0 :
            num_random_add = int(self.n_states * self.fraction_random_add())
            self.current_states[-num_random_add:] = self.get_new_states(num_random_add)
        for i in range(self.n_states):
            self.current_states[i] = self.evolve_state(self.current_states[i])

        return top_state_values

    def get_new_states(self, N):
        """Get N new random states

        Args:
            N (int): number of states to generatae

        Returns:
            array: set of states
        """
        states = []
        for i in range(N):
            new_state = self.environment.sample_state_space()
            states.append(new_state)
        return np.array(states)
    

def test_outputs():
    pass



if __name__ == '__main__':
    #env = gym.make('CartPole-v1')

    root_dir = "./genetic_real_2_50layers"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_layers = 10
    min_thickness = 0.0
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
        3: {
            'name': 'newmat',
            'n': 2.6,
            'a': 2,
            'alpha': 3.7e-6,
            'beta': 18e-6,
            'kappa': 32,
            'C': 2.1e6,
            'Y': 140e9,
            'prat': 0.43,
            'phiM': 8.44e-4
        },
    }

    thickness_options = [-0.1,-0.01,-0.001,0.0,0.001,0.01,0.1]

    env = CoatingStack(
        n_layers, 
        min_thickness, 
        max_thickness, 
        materials, 
        thickness_options=thickness_options)
    
    num_iterations = 2000
    statepool = StatePool(
        env, 
        n_states=5000, 
        states_fraction_keep = 0.3)

    filename = os.path.join(root_dir,'coating.png')
    scores = []
    max_scores = []
    eps_history = []
    n_steps = 0
    final_state = None
    
    n_mean_calc = 100
    for i in range(num_iterations):
        #statepool.fraction_keep_states = min(0.91 - 3*(i/num_iterations), 0.05)
        sort_state_values = statepool.evolve_step()
        score = np.mean(sort_state_values[:,1])
        scores.append(score)
        max_scores.append(sort_state_values[0,1])
        if i % 10 == 0:
            print('episode: ', i,'score %.5f ' % score)

    fig, ax = plt.subplots()
    ax.plot(scores[200:])
    fig.savefig(os.path.join(root_dir, "scores.png"))

    fig, ax = plt.subplots()
    ax.plot(scores[-400:])
    fig.savefig(os.path.join(root_dir, "scores_zoom.png"))

    fig, ax = plt.subplots()
    ax.plot(np.log(np.abs(max_scores))[200:])
    fig.savefig(os.path.join(root_dir, "max_scores.png"))

    fig, ax = plt.subplots()
    ax.plot(max_scores[-400:])
    fig.savefig(os.path.join(root_dir, "max_scores_zoom.png"))

    sorted_state_values = statepool.order_states()
    top_states = statepool.current_states[sorted_state_values[:10, 0].astype(int)]
    print(top_states)
    print(sorted_state_values[:10])

    top_state = top_states[0]

    print("-----------------------------")
    print(top_state)

    plotting.plot_coating(top_state, os.path.join(root_dir, "coating.png"))

    #print(top_state_value)

    """
    ### A little bit of index refining to bring together multiple layers of the same material 
    # Extract the second and third columns
    second_column = top_state_value[:, 1]
    third_column = top_state_value[:, 2]

    # Function to group consecutive indices based on matching values
    def group_consecutive(arr):
        groups = []
        current_group = [0]
        for i in range(1, len(arr)):
            if arr[i] == arr[current_group[-1]]:
                current_group.append(i)
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [i]
        if len(current_group) > 1:
            groups.append(current_group)
        return groups

    # Group consecutive indices in the second column
    groups_of_consecutive_indices_second_column = group_consecutive(second_column)

    # Group consecutive indices in the third column
    groups_of_consecutive_indices_third_column = group_consecutive(third_column)

    # Combine groups from both columns
    all_groups = groups_of_consecutive_indices_second_column + groups_of_consecutive_indices_third_column

    # Create a new array to store the results
    result_array = []

    # Keep track of indices that have been processed
    processed_indices = set()

    # Iterate through the original array
    for i in range(len(top_state_value)):
        if i in processed_indices:
            continue
        # Check if the current index is part of any group
        current_group = None
        for group in all_groups:
            if i in group:
                current_group = group
                break
        # If it's part of a group, sum the rows and add to result_array
        if current_group is not None:
            summed_row = np.sum(top_state_value[current_group, :], axis=0)
            summed_row[np.argmax(summed_row[1:])+1 ] = 1
            result_array.append(summed_row)
            processed_indices.update(current_group)
        # If it's not part of any group, add the row as it is
        else:
            result_array.append(top_state_value[i, :])

    # Convert result_array to a NumPy array
    result_array = np.array(result_array)
    
    ###
    print(result_array)
    


    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    depth_so_far = 0  # To keep track of where to plot the next bar
    colors = ["C0", "C1", "C2"]
    
    for i in range(len(result_array)):
        material_idx = np.argmax(result_array[i][1:]) 
        thickness = result_array[i][0]
        ax.bar(depth_so_far + thickness / 2, thickness, 
                width=thickness, 
                color=colors[material_idx])
        depth_so_far += thickness

    ax.set_xlim([0, depth_so_far * 1.01])
    ax.set_ylabel('Physical Thickness [nm]')
    ax.set_xlabel('Layer Position')
    ax.set_title('Generated Stack')

    fig.savefig(os.path.join(root_dir, "coating.png"))
    """

