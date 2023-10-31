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
from itertools import count
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.linear4(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        value = self.linear4(output)
        return value
    
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def plot_durations(episode_durations, root_dir):
    fig = plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    fig.savefig(os.path.join(root_dir, "loss.png"))


def trainIters(actor, critic, n_iters, optimiserA, optimiserC, device="cpu", root_dir="./"):
    #optimizerA = optim.Adam(actor.parameters())
    #optimizerC = optim.Adam(critic.parameters())


    episode_durations = []
    for iter in range(n_iters):
        state = env.reset().flatten()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            #env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            
            state = next_state.flatten()

            if done or i > 10000:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                episode_durations.append(i + 1)
                if i % 10 == 0:
                    plot_durations(episode_durations, root_dir) 
                break

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state.flatten())
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimiserA.zero_grad()
        optimiserC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimiserA.step()
        optimiserC.step()

        if iter % 20 == 0:
            print(f"Episode: {iter}, score: {rewards[-3:]}")

            torch.save({
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "actor_optimiser": optimiserA.state_dict(),
                "critic_optimiser": optimiserC.state_dict()
            },
            os.path.join(root_dir, "checkpoint_model.pt"))
    #env.close()

    print(next_state)
    print(next_value)

def test_model(actor,critic, n_starts):

    
    final_states = []
    final_rewards = []
    final_values = []
    for iter in range(n_starts):
        state = env.reset().flatten()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            #env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            
            state = next_state.flatten()

            if done or i > 10000:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                break

        final_states.append(state)
        final_rewards.append(reward)
        final_values.append(value)

    return final_states, final_rewards, final_values

if __name__ == '__main__':
    #env = gym.make('CartPole-v1')

    root_dir = "./actorcritic_real_2"
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
    
    num_iterations = 2000

    device = "cpu"

    actor = Actor(env.state_space_size, env.n_actions).to(device)
    critic = Critic(env.state_space_size, env.n_actions).to(device)

    optimiserA = optim.Adam(actor.parameters(), lr=1e-4)
    optimiserC = optim.Adam(critic.parameters(), lr=1e-4)

    trainIters(
        actor, 
        critic, 
        optimiserA=optimiserA, 
        optimiserC=optimiserC, 
        n_iters=num_iterations, 
        device=device,
        root_dir=root_dir)
    
    print("Testing")
    final_states, final_rewards, final_values = test_model(actor,critic, 5)

    print(final_states)
    print(final_rewards)
    print(final_values)
