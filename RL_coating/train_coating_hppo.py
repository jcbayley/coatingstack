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
from thermal_noise_environment import CoatingStack
#from simple_environment import CoatingStack
from itertools import count
from ppo import hppo
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

def training_loop(
        outdir, 
        env, 
        agent, 
        max_episodes=1000, 
        batch_size=256, 
        n_ep_train=10, 
        max_layers=4, 
        lower_bound=0,
        upper_bound=1, 
        useobs=False,
        beta_start=1.0,
        beta_end=0.001,
        beta_decay_length=None,
        lr_start=1e-3,
        lr_end=1e-5,
        lr_decay_length=4000):

    # Training loop
    rewards = []
    losses_pold = []
    losses_polc = []
    losses_val = []
    all_means = []
    all_stds = []
    all_mats = []
    max_reward = -np.inf
    max_state = None

    state_dir = os.path.join(outdir, "states")
    if not os.path.isdir(state_dir):
        os.makedirs(state_dir)

    for episode in range(max_episodes):
        if episode > 0:
            update_policy = True
        else:
            update_policy=True
        if episode > beta_decay_length:
            update_value=True
        else:
            update_value=True

        if beta_decay_length is not None:
            agent.beta = beta_start - (beta_start-beta_end)*episode/beta_decay_length

        if lr_decay_length is not None:
            new_lr = lr_start - (lr_start-lr_end)*episode/lr_decay_length
            agent.optimiser_discrete.learning_rate = new_lr
            agent.optimiser_continuous.learning_rate = new_lr
            agent.optimiser_value.learning_rate = new_lr

        states = []
        actions_discrete = []
        actions_continuous = []
        returns = []
        advantages = []
        for n in range(n_ep_train):
            state = env.reset()
            episode_reward = 0
            means = []
            stds = []
            mats = []
            t_rewards = []
            for t in range(100):
                # Select action
                fl_state = np.array([state.flatten(),])[0]
                obs = env.get_observation_from_state(state)
                if useobs:
                    obs = obs.flatten()
                else:
                    obs = fl_state
                action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_prob, c_means, c_std, value, entropy_discrete, entropy_continuous = agent.select_action(obs)

                action[1] = action[1]*(upper_bound - lower_bound) + lower_bound
                """
                if t == 0:
                    action[0] = 2
                if t == 1:
                    action[0] = 1
                if t == 2:
                    action[0] = 2
                if t == 3:
                    action[0] = 1
                """
                # Take action and observe reward and next state
                next_state, reward, done, finished, _, full_action = env.step(action)

                t_rewards.append(reward)
                agent.replay_buffer.update(
                    actiond,
                    actionc,
                    obs,
                    log_prob_discrete,
                    log_prob_continuous,
                    reward,
                    value,
                    done,
                    entropy_discrete,
                    entropy_continuous
                )
                #log_prob, state_value, entropy = agent.evaluate(fl_state, action[0], action[1])

                means.append(c_means.detach().numpy())
                stds.append(c_std.detach().numpy())
                mats.append(d_prob.detach().numpy().tolist()[0])
                fl_next_state = next_state.flatten()
                # Store transition in replay buffer
                #agent.policy.rewards.append(reward)

                # Update state and episode rewardz
                state = next_state
                episode_reward += reward

                # Check if episode is done
                if done or finished:
                    break

            if episode_reward > max_reward:
                max_reward = episode_reward
                max_state = state

            returns = agent.get_returns(t_rewards)
            agent.replay_buffer.update_returns(returns)
            

        all_means.append(means)
        all_stds.append(stds)
        all_mats.append(mats)

        if episode > 10:
            loss1, loss2, loss3 = agent.update(update_policy=update_policy, update_value=update_value)
            losses_pold.append(loss1)
            losses_polc.append(loss2)
            losses_val.append(loss3)
        rewards.append(episode_reward)

        if episode % 20 == 0 and episode !=0 :
            reward_fig, reward_ax = plt.subplots()
            window_size = 20
            downsamp_rewards = np.mean(np.reshape(rewards[:int((len(rewards)//window_size)*window_size)], (-1,window_size)), axis=1)
            reward_ax.plot(np.arange(episode+1), rewards)
            reward_ax.plot(np.arange(episode).reshape(-1,window_size)[:,0], downsamp_rewards)
            reward_fig.savefig(os.path.join(outdir, "running_rewards.png"))


            loss_fig, loss_ax = plt.subplots(nrows=3)
            loss_ax[0].plot(losses_pold)
            loss_ax[1].plot(losses_polc)
            loss_ax[2].plot(losses_val)
            loss_ax[0].set_ylabel("Policy discrete loss")
            loss_ax[1].set_ylabel("Policy continuous loss")
            loss_ax[2].set_ylabel("Value loss")
            loss_ax[2].set_yscale("log")
            loss_fig.savefig(os.path.join(outdir, "running_losses.png"))

            """
            n_layers = max_layers
            loss_fig, loss_ax = plt.subplots(nrows = n_layers)
            all_means2 = pad_lists(all_means, np.nan, n_layers)
            all_stds2 = pad_lists(all_stds, np.nan, n_layers)
            for i in range(n_layers):
                loss_ax[i].plot(all_means2[:,i])
                loss_ax[i].plot(all_stds2[:,i])
            loss_fig.savefig(os.path.join(outdir, "running_means.png"))
            """
            
            n_layers = max_layers
            loss_fig, loss_ax = plt.subplots(nrows = n_layers)
            all_mats2 = pad_lists(all_mats, [0.0,0.0,0.0], n_layers)
            #all_mats2 = all_mats2[:,:,:]
            for i in range(n_layers):
                for mind in range(len(all_mats2[0, i])):
                    loss_ax[i].scatter(np.arange(len(all_mats2)), np.ones(len(all_mats2))*mind, s=10*all_mats2[:,i,mind])
            loss_fig.savefig(os.path.join(outdir, "running_mats.png"))
            
            # Print episode information
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

            if episode % 100 == 0:
                stfig = plot_state(state)
                stfig.savefig(os.path.join(state_dir, f"episode_{episode}.png"))

    print("Max_state: ", max_reward)
    print(max_state)

    return rewards

if __name__ == "__main__":
    root_dir = "./tmm_obs_sparse_test_hppo_8layer"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_layers = 8
    min_thickness = 1e-9
    max_thickness = 300e-9
    #min_thickness = 0.1
    #max_thickness = 1
    load_model = False
    useobs=True

    materials = {
        0:{
            'name' : 'air', 
            'n'    : 1,
            'a'    : 0,
            'alpha': np.NaN,
            'beta' : np.NaN,
            'kappa': np.NaN,
            'C'    : np.NaN,
            'Y'    : np.NaN,
            'prat' : np.NaN,
            'phiM' : np.NaN,
            'k'    : 0
            },
        1: {
            'name' : 'SiO2',
            'n'    : 1.44,
            'a'    : 0,
            'alpha': 0.51e-6,
            'beta' : 8e-6,
            'kappa': 1.38,
            'C'    : 1.64e6,
            'Y'    : 72e9,
            'prat' : 0.17,
            'phiM' : 4.6e-5,
            'k'    : 1
        },
        2: {
            'name' : 'ta2o5',
            'n'    : 2.07,
            'a'    : 2,
            'alpha': 3.6e-6,
            'beta' : 14e-6,
            'kappa': 33,
            'C'    : 2.1e6,
            'Y'    : 140e9,
            'prat' : 0.23,
            'phiM' : 2.44e-4,
            'k'    :1
        },
    }

    thickness_options = [0.0]

    env = CoatingStack(
        n_layers, 
        min_thickness, 
        max_thickness, 
        materials)

    
    num_iterations = 2000

    device = "cpu"

    insize = env.obs_space_size if useobs==True else env.state_space_size

    agent = hppo.PPO(
            insize,
            env.n_materials, 
            env.n_materials,
            hidden_size=32,
            lr_policy=1e-3, 
            lr_value=1e-3,
            lower_bound=0,
            upper_bound=1,
            n_updates=5,
            beta=0.1
            )
    
    
    rewards = training_loop(
        root_dir, 
        env, 
        agent, 
        max_episodes=num_iterations,
        n_ep_train=64,
        max_layers=n_layers,
        useobs=useobs,
        beta_start=0.5,
        beta_end=0.1,
        beta_decay_length=1000,
        lr_start=1e-3,
        lr_end=1e-3,
        lr_decay_length=3000,
        upper_bound=max_thickness,
        lower_bound=min_thickness)

    episode_rewards = []
    episode_returns = []
    states = []
    for i in range(100):
        state = env.reset()
        episode_return = 0
        for t in range(100):
            # Select action
            #action = agent.select_action(fl_state)
            fl_state = np.array([state.flatten(),])
            obs = env.get_observation_from_state(state)
            if useobs:
                obs = obs.flatten()
            else:
                obs = fl_state
            action, actiond, actionc, log_prob_d, log_prob_c, d_prob, c_means, c_std, value, entropy_d, entropy_c = agent.select_action(obs)

            action[1] = action[1]*(max_thickness - min_thickness) + min_thickness
            # Take action and observe reward and next state
            next_state, reward, done, finished, _, full_action = env.step(action)


            # Update state and episode reward
            state = next_state
            episode_return += reward

            if done or finished:
                break
        
        episode_rewards.append(reward)
        episode_returns.append(episode_return)
        states.append(state)

    maxind = -1#np.argmax(episode_returns)
    print(states[maxind])
    print("Return: ", episode_rewards[maxind])
    print("Reward: ", episode_rewards[maxind])

    maxind = np.argmax(episode_returns)
    print(states[maxind])
    print("Max Return: ", episode_rewards[maxind])
    print("Max Reward: ", episode_rewards[maxind])

    fig, ax = plt.subplots()
    ax.hist(episode_rewards)
    fig.savefig(os.path.join(root_dir, "./return_hist.png"))

    thickness1 = 1064e-9 /(4*materials[1]["n"])
    thickness2 = 1064e-9 /(4*materials[2]["n"])
    print("thickness", thickness1, thickness2)
    max_state = np.array([[thickness1, 0, 1, 0],
                          [thickness2, 0, 0, 1],
                          [thickness1, 0, 1, 0],
                          [thickness2, 0, 0, 1]])
    
    max_reward = env.compute_state_value(max_state)
    max_reward_ligo = env.compute_state_value_ligo(max_state)
    max_reward_ligo2 = env.compute_state_value_ligo(max_state[::-1])

    max_state1 = np.array([[thickness2*1e9, 0, 0, 1],
                          [thickness1*1e9, 0, 1, 0],
                          [thickness2*1e9, 0, 0, 1],
                          [thickness1*1e9, 0, 1, 0]])
    
    max_reward1 = env.compute_state_value(max_state1)
    max_reward1_ligo = env.compute_state_value_ligo(max_state1)
    max_reward1_ligo2 = env.compute_state_value_ligo(max_state1[::-1])

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

    print("ligocomp1", max_reward1, max_reward1_ligo, max_reward1_ligo2)
    print("ligocomp0", max_reward, max_reward_ligo, max_reward_ligo2)


