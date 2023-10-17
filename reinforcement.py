import torch
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import random
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Environment():

    def __init__(self, max_layers, min_thickness, max_thickness, materials, variable_layers=False):
        self.variable_layers = variable_layers
        self.max_layers = max_layers
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.materials = materials
        self.n_materials = len(materials)

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
    
    def sample_action_space(self, ):
        """sample from the action space

        Returns:
            _type_: _description_
        """
        layer = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(0,self.max_layers))), num_classes=self.max_layers)

        material_change = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(self.n_materials))), num_classes=self.n_materials)
        thickness_range = (self.max_thickness - self.min_thickness)/60
        thickness_change = np.random.uniform(-thickness_range, thickness_range)
        #print(layer, material_change, thickness_change)
        return np.concatenate([layer, material_change, [thickness_change]])

    def compute_reward(self,state):
        """_summary_

        Args:
            state (_type_): _description_
        """

        reward = 0
        #print(len(state))
        for i,layer in enumerate(state):
            if i == 0:
                continue
            else:
                last_material = self.materials[torch.argmax(state[i-1][1:]).item()]
                current_material = self.materials[torch.argmax(state[i][1:]).item()]
                last_thickness = state[i-1][0]
                current_thickness = state[i][0]
                refract_diff = current_material["n"]/current_thickness - last_material["n"]/last_thickness

                #print("rdiff", refract_diff)
                reward += refract_diff
        
        return reward

    def get_new_state(self, current_states, action):
        """new state is the current action choice

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
 
        layer = torch.argmax(action[:,:self.max_layers]).item()
        material = action[:,self.max_layers:self.max_layers+self.n_materials]
        thickness_change = action[:,self.max_layers+self.n_materials:].item()
        current_states[:,layer][0] += thickness_change
        current_states[:,layer][1:] = material

        return current_states


    def step(self, action):
        """action[0] - thickness
           action[1:N] - material probability

        Args:
            action (_type_): _description_
        """
        
        terminated = False
        if torch.any(action[:,2] < self.min_thickness) or torch.any(action[:,2] > self.max_thickness):
            print("out of thickness bounds")
            terminated = True
        
        new_state = self.get_new_state(self.current_state, action)
        reward = self.compute_reward(new_state)
        self.length += 1
        print("Reward", reward.item())
        #self.print_state()

        return new_state, reward, terminated


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(torch.nn.Module):

    def __init__(self, n_observations,  n_material, n_layers):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.material_out = torch.nn.Linear(128, n_material)
        self.layer_out = torch.nn.Linear(128, n_layers)
        self.thickness_out = torch.nn.Linear(128, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        siglayers = torch.nn.functional.softmax(self.material_out(x))
        sigmaterial = torch.nn.functional.softmax(self.layer_out(x))
        thick = self.thickness_out(x)
        x = torch.cat([thick, sigmaterial, siglayers], dim=1)
        return x

def select_action(state, steps_done, env):
    """_summary_

    Args:
        state (_type_): _description_
        steps_done (_type_): _description_
        env (_type_): _description_

    Returns:
        _type_: _description_
    """
    sample = np.random.uniform()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            print("Network")
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(torch.flatten(state, start_dim=1))#.max(1)[1].view(1, 1)
    else:
        return torch.tensor([env.sample_action_space()], device=device, dtype=torch.long)


def optimize_model(memory, batch_size, policy_net, target_net, optimiser):
    """_summary_

    Args:
        memory (_type_): _description_
        batch_size (_type_): _description_
        policy_net (_type_): _description_
        target_net (_type_): _description_
        optimiser (_type_): _description_
    """
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    print("loss", loss)
    # Optimize the model
    optimiser.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimiser.step()


if __name__ == "__main__":
    batch_size = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    device = "cpu"

    n_layers = 3
    num_episodes = 10
    min_thickness = 1
    max_thickness = 4

    materials = {
        0:{"n": 2},
        1:{"n": 7},
        2:{"n": 5},
    }

    env = Environment(n_layers, min_thickness, max_thickness, materials,)

    # Get the number of state observations
    state = env.reset()
    n_observations = env.state_space_size

    # define number of actions to be made
    #  choose thickness, choose layer, then choose material (onehot)
    n_actions = 1 + env.max_layers + len(materials)

    policy_net = DQN(n_observations, env.max_layers, len(materials)).to(device)
    target_net = DQN(n_observations, env.max_layers, len(materials)).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimiser = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    truncated = 200

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        print(f"Episode: {i_episode}")
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        print(len(memory))
        for t in count():
            action = select_action(state, t, env)
            #print(np.shape(action))
            observation, reward, terminated = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or (t > truncated)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, batch_size, policy_net, target_net, optimiser)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                #episode_durations.append(t + 1)
                # plot_durations()
                break