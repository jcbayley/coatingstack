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
from itertools import count
from torch.distributions import Categorical
import plotting

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, n_hidden=2):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.linearin = nn.Linear(self.state_size, hidden_size)
        #self.linear2 = nn.Linear(128, 256)
        for i in range(self.n_hidden):
            setattr(self, f"linear{i}", nn.Linear(hidden_size, hidden_size))
        self.linearout = nn.Linear(hidden_size, self.action_size)

    def forward(self, state):
        output = F.relu(self.linearin(state))
        for i in range(self.n_hidden):
            output = F.relu(getattr(self, f"linear{i}")(output))
        #output = F.relu(self.linear2(output))
        output = self.linearout(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, n_hidden=2):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.linearin = nn.Linear(self.state_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, hidden_size)
        for i in range(self.n_hidden):
            setattr(self, f"linear{i}", nn.Linear(hidden_size, hidden_size))
        self.linearout = nn.Linear(hidden_size, 1)

    def forward(self, state):
        output = F.relu(self.linearin(state))
        #output = F.relu(self.linear2(output))
        for i in range(self.n_hidden):
            output = F.relu(getattr(self, f"linear{i}")(output))
        value = self.linearout(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def plot_durations(episode_durations, root_dir):
    fig, ax  = plt.subplots()
    durations_t = torch.FloatTensor(episode_durations)
    ax.set_title('Training...')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration')
    ax.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax.plot(means.numpy())

    #plt.pause(0.001)  # pause a bit so that plots are updated

    fig.savefig(os.path.join(root_dir, "loss.png"))

def plot_rewards(iter, rewards, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.figure()

    ax.set_xlabel("iteration")
    ax.set_ylabel("reward")

    steps = np.arange(len(rewards))
    if len(np.shape(rewards)) > 1:
        rewards = np.array(rewards)[:,0]
    else:
        rewards = np.array(rewards)
    #okpos = rewards < -1
    #print(np.shape(okpos), np.shape(rewards), np.shape(steps))
    ax.plot(steps, rewards, label=f"Iteration: {iter}")
    ax.set_ylim([-10,max(rewards) + 1])
    ax.legend()

    plt.close(fig)

    return fig

def plot_values(iter, values, pred_values, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots()

    ax.set_xlabel("iteration")
    ax.set_ylabel("reward")

    steps = np.arange(len(values))
    if len(np.shape(values)) > 1:
        values = np.array(values)[:,0]
    else:
        values = np.array(values)
    if len(np.shape(pred_values)) > 1:
        pred_values = np.array(pred_values)[:,0]
    else:
        pred_values = np.array(pred_values)
    #okpos = rewards < -1
    #print(np.shape(okpos), np.shape(rewards), np.shape(steps))
    ax.plot(steps, values - pred_values, label=f"Values: {iter}")
    #ax.plot(steps, pred_values, label=f"Pred Values: {iter}")
    #ax.set_ylim([-1,np.max([values, pred_values]) + 1])
    ax.legend()

    plt.close(fig)

    return fig

def plot_score(end_scores, max_scores, fname=None):

    fig, ax = plt.subplots()
    ax.plot(max_scores, label="max_scores", marker=".", ls="none")
    ax.plot(end_scores, label="end_scores", marker=".", ls="none")
    ax.set_xlabel("episode")
    ax.set_ylabel("score")
    ax.set_ylim([-10, max(max_scores) + 1])

    if fname is not None:
        fig.savefig(fname)


def trainIters(actor, critic, n_iters, optimiserA, optimiserC, device="cpu", root_dir="./"):
    #optimizerA = optim.Adam(actor.parameters())
    #optimizerC = optim.Adam(critic.parameters())


    episode_durations = []
    episode_median_scores = []
    episode_end_scores = []
    episode_max_scores = []
    fig, ax = plt.subplots()
    figval, axval = plt.subplots()
    figvals, axvals = plt.subplots()
    figret, axret = plt.subplots()
    for iter in range(n_iters):
        state = env.reset().flatten()
        log_probs = []
        values = []
        rewards = []
        state_vals = []
        masks = []
        entropy = 0
        env.reset()
        max_state = -100

        for i in count():
            #env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            next_state, reward, done, new_value = env.step(action.cpu().numpy(), max_state)
            
            if new_value > max_state:
                max_state = new_value

            state_vals.append(new_value)

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            
            state = next_state.flatten()

            if done or i > 5000:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                episode_durations.append(i + 1) 
                break

        
        episode_end_scores.append(state_vals[-1])
        episode_max_scores.append(np.max(state_vals))

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

        if iter % 100 == 0 and iter > 0:
            plot_durations(episode_durations, root_dir) 
            plot_rewards(iter, rewards, fig, ax)
            fig.savefig(os.path.join(root_dir, "rewards.png"))
            plot_rewards(iter, state_vals, figval, axval)
            figval.savefig(os.path.join(root_dir, "state_values.png"))

            #plot_values(iter, state_vals, values.detach().numpy(), fig=figvals, ax=axvals)
            #figvals.savefig(os.path.join(root_dir, "values_state.png"))

            plot_values(iter, returns, values.detach().numpy(), fig=figret, ax=axret)
            figret.savefig(os.path.join(root_dir, "values.png"))

            plot_score(episode_end_scores, episode_max_scores, fname=os.path.join(root_dir, "running_scores.png"))


        if iter % 100 == 0:
            print(f"Episode: {iter}, score: {np.mean(rewards)}, {rewards[0]} ,{rewards[-1]}")

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

def test_model(actor,critic, n_starts, n_layers=5, root_dir="./"):
    """run a number of tests of the model to see what optimal state is

    Args:
        actor (_type_): _description_
        critic (_type_): _description_
        n_starts (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    final_states = []
    final_rewards = []
    final_values = []
    final_state_values = []
    for iter in range(n_starts):
        state = env.reset().flatten()
        log_probs = []
        values = []
        rewards = []
        state_vals = []
        masks = []
        all_states = []
        entropy = 0
        env.reset()

        for i in count():
            #env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            # get most likely action each time
            #action = torch.argmax(dist.log_prob(torch.arange(env.n_actions)))
            next_state, reward, done, new_value = env.step(action.cpu().numpy(), 0)

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            state_vals.append(new_value)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1-done)
            all_states.append(next_state)
            
            state = next_state.flatten()

            if done or i > 5000:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                break

        final_states.append(all_states)
        final_rewards.append(rewards)
        final_values.append(value)
        final_state_values.append(state_vals)

    for i in range(n_starts):
        max_ind = np.argmax(final_rewards[i])
        max_state = final_states[i][max_ind]
        print(np.array(max_state).reshape(n_layers, -1))
        print(final_rewards[i][max_ind])
        print(final_rewards[i][-1])

        plotting.plot_coating(max_state, os.path.join(root_dir, f"coating_{i}.png"))

    fig, ax = plt.subplots()
    for i in range(n_starts):
        ax.plot(final_rewards[i])

    ax.set_ylim([-1,float(np.max(final_rewards)) + 1])
    
    fig.savefig(os.path.join(root_dir, "test_rewards.png"))

    fig, ax = plt.subplots()
    for i in range(n_starts):
        ax.plot(final_state_values[i])

    ax.set_ylim([-1,max(final_state_values) + 1])
    
    fig.savefig(os.path.join(root_dir, "test_state_values.png"))

    fig, ax = plt.subplots()
    for i in range(n_starts):
        ax.plot(final_values[i])

    ax.set_ylim([-1,max(final_values) + 1])
    
    fig.savefig(os.path.join(root_dir, "test_values.png"))
    

    return final_states, final_rewards, final_values

if __name__ == '__main__':
    #env = gym.make('CartPole-v1')

    root_dir = "./actorcritic_updated_plot"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_layers = 5
    min_thickness = 0.01
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

    thickness_options = [0.0]

    env = CoatingStack(
        n_layers, 
        min_thickness, 
        max_thickness, 
        materials, 
        thickness_options=thickness_options)
    
    num_iterations = 1000

    device = "cpu"

    actor = Actor(env.state_space_size, env.n_actions, hidden_size=256, n_hidden=3).to(device)
    critic = Critic(env.state_space_size, env.n_actions, hidden_size=256, n_hidden=3).to(device)

    #actor = Actor(env.state_space_size, env.n_actions).to(device)
    #critic = Critic(env.state_space_size, env.n_actions).to(device)

    optimiserA = optim.Adam(actor.parameters(), lr=1e-3)
    optimiserC = optim.Adam(critic.parameters(), lr=1e-3)

    trainIters(
        actor, 
        critic, 
        optimiserA=optimiserA, 
        optimiserC=optimiserC, 
        n_iters=num_iterations, 
        device=device,
        root_dir=root_dir)
    
    print("Testing")
    n_examples = 5
    with torch.no_grad():
        final_states, final_rewards, final_values = test_model(
            actor,
            critic, 
            n_examples,
            n_layers,
            root_dir)

 
