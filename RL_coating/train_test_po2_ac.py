import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt
import copy 
#from thermal_noise_environment import CoatingStack
#from simple_environment import CoatingStack
from itertools import count
from ppo import po_2_ac, po, ppo_2, ppo_3_ac
import itertools

def pad_lists(list_of_lists, padding_value=0, max_length=3):
    if max_length is None:
        max_length = max(len(lst) for lst in list_of_lists)
    padded_lists = []
    for lst in list_of_lists:
        t_lst = []
        for l in lst:
            t_lst.append(l)

        if len(t_lst) < max_length:
            diff = int(max_length - len(t_lst))
            for i in range(diff):
                t_lst.append(padding_value)

        
        padded_lists.append(t_lst)

    #padded_lists = [lst + [padding_value] * (max_length - len(lst)) for lst in list_of_lists]
    return np.array(padded_lists)

def plot_state(state):
    thicknesses = state[:,0]
    materials = np.argmax(state[:,1:], axis=1)

    colors = ["k", "C0", "C1"]

    fig,ax = plt.subplots()
    scoord = 0
    for i in range(len(state)):
        ax.bar(scoord, 1, color=colors[materials[i]], width=thicknesses[i], align="edge")
        scoord += thicknesses[i]

    return fig


def training_loop(outdir, env, agent, max_episodes=1000, batch_size=256, n_ep_train=10, max_layers=4, lower_bound=0,upper_bound=1):

    # Training loop
    rewards = []
    losses_pol = []
    losses_val = []
    all_means = []
    all_stds = []
    all_mats = []
    max_reward = -np.inf
    max_state = None

    state_dir = os.path.join(outdir, "states")
    if not os.path.isdir(state_dir):
        os.makedirs(state_dir)

    update_policy = True
    for episode in range(max_episodes):

        states = []
        actions_discrete = []
        actions_continuous = []
        returns = []
        advantages = []
        for n in range(n_ep_train):
            state, _ = env.reset()
            episode_reward = 0
            means = []
            stds = []
            mats = []
            t_rewards = []
            for t in range(100):
                # Select action
                fl_state = state.flatten()
                
                action, actiond, actionc, log_prob, d_prob, c_means, c_std, value, entropy = agent.select_action(fl_state)

                action[1] = action[1]*(upper_bound - lower_bound) + lower_bound


                #if episode < 1:
                #    action = [np.random.randint(0,3), np.random.uniform(upper_bound, lower_bound)]

                # Take action and observe reward and next state
      
                next_state, reward, done, _, _ = env.step(int(action[0].item()))

                t_rewards.append(reward)
                agent.replay_buffer.update(
                    actiond,
                    actionc,
                    fl_state,
                    log_prob,
                    reward,
                    value,
                    done,
                    entropy
                )
                #log_prob, state_value, entropy = agent.evaluate(fl_state, action[0], action[1])

                means.append(c_means.item())
                stds.append(c_std.item())
                mats.append(d_prob.detach().numpy().tolist()[0])
                fl_next_state = next_state.flatten()
                # Store transition in replay buffer
                #agent.policy.rewards.append(reward)

                # Update state and episode rewardz
                state = next_state
                episode_reward += reward

                # Check if episode is done
                if done:
                    #agent.replay_buffer.next_episode()
                    break
                
            returns = agent.get_returns(t_rewards)
            agent.replay_buffer.update_returns(returns)

            if episode_reward > max_reward:
                max_reward = episode_reward
                max_state = state
            

        all_means.append(means)
        all_stds.append(stds)
        all_mats.append(mats)

        if episode > 10:
            if episode < 0:
                update_policy = False
            else:
                update_policy = True
            loss1, loss2 = agent.update(update_policy)
            losses_pol.append(loss1)
            losses_val.append(loss2)
        rewards.append(episode_reward)

        if episode % 20 == 0 and episode !=0 :
            reward_fig, reward_ax = plt.subplots()
            window_size = 20
            downsamp_rewards = np.mean(np.reshape(rewards[:int((len(rewards)//window_size)*window_size)], (-1,window_size)), axis=1)
            reward_ax.plot(np.arange(episode+1), rewards)
            reward_ax.plot(np.arange(episode).reshape(-1,window_size)[:,0], downsamp_rewards)
            reward_fig.savefig(os.path.join(outdir, "running_rewards.png"))


            loss_fig, loss_ax = plt.subplots(nrows=2)
            loss_ax[0].plot(losses_pol)
            loss_ax[1].plot(losses_val)
            loss_ax[0].set_ylabel("Policy loss")
            loss_ax[1].set_ylabel("Value loss")
            loss_ax[1].set_yscale("log")
            loss_fig.savefig(os.path.join(outdir, "running_losses.png"))
            """
            n_layers = max_layers
            loss_fig, loss_ax = plt.subplots(nrows = n_layers+1)
            all_means2 = pad_lists(all_means, np.nan, n_layers)
            all_stds2 = pad_lists(all_stds, np.nan, n_layers)
            for i in range(n_layers):
                loss_ax[i].plot(all_means2[:,i])
                loss_ax[i].plot(all_stds2[:,i])
            loss_fig.savefig(os.path.join(outdir, "running_means.png"))

            
            n_layers = 4
            loss_fig, loss_ax = plt.subplots(nrows = n_layers)
            all_mats2 = pad_lists(all_mats, [0.0,0.0,0.0], n_layers)
            #all_mats2 = all_mats2[:,:,:]
            for i in range(n_layers):
                for mind in range(len(all_mats2[0, i])):
                    loss_ax[i].scatter(np.arange(len(all_mats2)), np.ones(len(all_mats2))*mind, s=10*all_mats2[:,i,mind])
            loss_fig.savefig(os.path.join(outdir, "running_mats.png"))
            """
            # Print episode information
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

        #if episode % 100 == 0:
        #    stfig = plot_state(state)
        #    stfig.savefig(os.path.join(state_dir, f"episode_{episode}.png"))

    print("Max_state: ", max_reward)
    print(max_state)

    return rewards

if __name__ == "__main__":
    root_dir = "./cartpole_test_po2_acad"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_layers = 2
    

    env = gym.make("CartPole-v1")
    #env = gym.make("CarRacing-v2", domain_randomize=True)
    print(env.action_space.shape)
    print(env.observation_space.shape)

    n_discrete = 2
    min_thickness = 0
    max_thickness = 1
    
    num_iterations = 1000

    device = "cpu"
    """
    agent = po_2_ac.PO(env.observation_space.shape[0],
                    n_discrete, 
                    1,
                    hidden_size=64,
                    lr_policy=1e-3, 
                    lr_value=2e-3,
                    lower_bound=0,
                    upper_bound=1,
                    n_updates=1,
                    )
    """
    agent = ppo_3_ac.PPO(
            env.observation_space.shape[0],
                    n_discrete, 
                    1,
                    hidden_size=64,
                    lr_policy=1e-3, 
                    lr_value=2e-3,
                    lower_bound=0,
                    upper_bound=1,
                    n_updates=1,
                    )
    """
    
    agent = po.PO(env.observation_space.shape[0],
                    n_discrete, 
                    1,
                    hidden_size=64,
                    lr=1e-3,
                    lower_bound=0,
                    upper_bound=1,
                    n_updates=1,
                    )
    """
    rewards = training_loop(
        root_dir, 
        env, 
        agent, 
        max_episodes=num_iterations,
        n_ep_train=32,
        max_layers=n_layers,
        lower_bound=min_thickness,
        upper_bound=max_thickness)

    episode_rewards = []
    episode_returns = []
    episode_values = []
    states = []
    for i in range(100):
        state, _ = env.reset()
        episode_return = 0
        for t in range(100):
            # Select action
            fl_state = state.flatten()
            #action = agent.select_action(fl_state)
            action, actiond, actionc, log_prob, d_prob, c_means, c_std, value, entropy  = agent.select_action(fl_state)

            action[1] = action[1]*(max_thickness - min_thickness) + min_thickness
            # Take action and observe reward and next state
            next_state, reward, done, _ = env.step(action[0].item())


            # Update state and episode reward
            state = next_state
            episode_return += reward

            if done:
                break
        episode_values.append(value)
        
        episode_rewards.append(reward)
        episode_returns.append(episode_return)
        states.append(state)

    maxind = -1#np.argmax(episode_returns)
    print("Final test state :")
    print(states[maxind])
    print("Return: ", episode_returns[maxind], episode_values[maxind])
    print("Reward: ", episode_rewards[maxind])

    print("Max test State")
    maxind = np.argmax(episode_returns)
    print(states[maxind])
    print("Max Return: ", episode_returns[maxind], episode_values[maxind])
    print("Max Reward: ", episode_rewards[maxind])

    fig, ax = plt.subplots()
    ax.hist(episode_rewards)
    fig.savefig(os.path.join(root_dir, "./return_hist.png"))

    thickness1 = 1064e-9 /(4*materials[1]["n"]) 
    thickness2 = 1064e-9 /(4*materials[2]["n"])
    print("thickness", thickness1, thickness2)
     
    max_state = np.array([[thickness1, 0, 0, 1],
                          [thickness2, 0, 1, 0]])
    
    max_reward = env.compute_state_value(max_state)

    max_state1 = np.array([[thickness2*1e9, 0, 0, 1],
                          [thickness1*1e9, 0, 1, 0]])
    
    max_reward1 = env.compute_state_value(max_state1)

    max_state2 = np.array([[thickness2, 0, 0, 1],
                          [thickness2, 0, 0, 1]])
    
    max_reward2 = env.compute_state_value(max_state2)

    max_state3 = np.array([[thickness1, 0, 1, 0],
                          [thickness1, 0, 1, 0]])
    
    max_reward3 = env.compute_state_value(max_state3)
    print("optimal reward", max_reward, max_reward1)
    print("nonopt: ", max_reward2, max_reward3)
    """
    max_state = np.array([[thickness1, 0, 1, 0],
                          [thickness2, 0, 0, 1],
                          [thickness1, 0, 1, 0],
                          [thickness2, 0, 0, 1]])
    
    max_reward = env.compute_state_value(max_state)

    max_state1 = np.array([[thickness2*1e9, 0, 0, 1],
                          [thickness1*1e9, 0, 1, 0],
                          [thickness2*1e9, 0, 0, 1],
                          [thickness1*1e9, 0, 1, 0]])
    
    max_reward1 = env.compute_state_value(max_state1)

    max_state2 = np.array([[thickness2*1e9, 0, 0, 1],
                          [thickness1, 0, 1, 0],
                          [thickness2*1e9, 0, 0, 1],
                          [thickness1, 0, 1, 0]])
    
    max_reward2 = env.compute_state_value(max_state2)

    max_state3 = np.array([[thickness1, 0, 1, 0],
                          [thickness1, 0, 1, 0],
                          [thickness1, 0, 1, 0],
                          [thickness1, 0, 1, 0]])
    
    max_reward3 = env.compute_state_value(max_state3)
    print("optimal reward", max_reward, max_reward1)
    print("nonopt: ", max_reward2, max_reward3)

    """


