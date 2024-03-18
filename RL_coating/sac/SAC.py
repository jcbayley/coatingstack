
import torch
from torch.distributions import Normal
import numpy as np
from truncated_normal import TruncatedNormalDist
from .networks import ContinuousPolicyNetwork, QNetwork

class SAC(object):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"
    def __init__(
            self, 
            state_dim, 
            continuous_action_dim,
            hidden_size,
            alpha = 0.1,
            gamma = 0.99,
            tau = 0.005,
            lr=1e-4,
            gradient_clipping_norm = None,
            gaussian_lower_bound=0, 
            gaussian_upper_bound=1,
            auto_tune_entropy=True,
            device="cpu"):
        
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.gradient_clipping_norm = gradient_clipping_norm
        self.continuous_action_dim = continuous_action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.device=device
        self.gaussian_lower_bound = gaussian_lower_bound
        self.gaussian_upper_bound = gaussian_upper_bound

        self.EPSILON = 1e-6
        
        self.actor = ContinuousPolicyNetwork(state_dim, continuous_action_dim, hidden_size, gaussian_lower_bound, gaussian_upper_bound)
        self.critic1 = QNetwork(state_dim,  continuous_action_dim, hidden_size)
        self.critic2 = QNetwork(state_dim, continuous_action_dim, hidden_size)
        self.target_critic1 = QNetwork(state_dim, continuous_action_dim, hidden_size)
        self.target_critic2 = QNetwork(state_dim, continuous_action_dim, hidden_size)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = torch.optim.Adam(list(self.critic1.parameters()), lr=self.lr)
        self.critic2_optimizer = torch.optim.Adam(list(self.critic2.parameters()), lr=self.lr)


        self.automatic_entropy_tuning = auto_tune_entropy
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.continuous_action_dim).to(self.device)).item() # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)
        


    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """_summary_

        Args:
            optimizer (_type_): _description_
            network (_type_): _description_
            loss (_type_): _description_
            clipping_norm (_type_, optional): _description_. Defaults to None.
            retain_graph (bool, optional): _description_. Defaults to False.
        """
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def pick_action(self, state, eval_ep=False):
        """_summary_

        Args:
            eval_ep (_type_): _description_
            state (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        if eval_ep: 
            action = self.actor_pick_action(state=state, eval=True)
        else: 
            action = self.actor_pick_action(state=state)
        return action

    def actor_pick_action(self, state=None, eval=False):
        """_summary_

        Args:
            state (_type_, optional): _description_. Defaults to None.
            eval (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: 
            state = state.unsqueeze(0)
        if eval == False: 
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """_summary_

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        actor_mean, actor_log_std = self.actor(state)
        #mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = actor_log_std.exp()
        normal = TruncatedNormalDist(actor_mean, std, self.gaussian_lower_bound, self.gaussian_upper_bound)
        x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + self.EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(actor_mean)


    def learn(self, batch):
        """_summary_

        Args:
            batch (_type_): _description_
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.update_critic_parameters(qf1_loss, qf2_loss)

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning: alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else: alpha_loss = None
        self.update_actor_parameters(policy_loss, alpha_loss)

        return qf1_loss, qf2_loss, policy_loss


    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """_summary_

        Args:
            state_batch (_type_): _description_
            action_batch (_type_): _description_
            reward_batch (_type_): _description_
            next_state_batch (_type_): _description_
            mask_batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.target_critic1(next_state_batch, next_state_action)
            qf2_next_target = self.target_critic2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.gamma * (min_qf_next_target)
        #print("state", state_batch.size(), next_state_batch.size())
        #print("action", action_batch.size(), next_state_action.size())
        qf1 = self.critic1(state_batch, action_batch)
        qf2 = self.critic2(state_batch, action_batch)
        qf1_loss = torch.nn.functional.mse_loss(qf1, next_q_value)
        qf2_loss = torch.nn.functional.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """_summary_

        Args:
            state_batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic1(state_batch, action)
        qf2_pi = self.critic2(state_batch, action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """_summary_

        Args:
            log_pi (_type_): _description_

        Returns:
            _type_: _description_
        """
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """_summary_

        Args:
            critic_loss_1 (_type_): _description_
            critic_loss_2 (_type_): _description_
        """
        self.take_optimisation_step(
            self.critic1_optimizer,
            self.critic1, 
            critic_loss_1,
            self.gradient_clipping_norm)
        self.take_optimisation_step(
            self.critic2_optimizer, 
            self.critic2, 
            critic_loss_2,
            self.gradient_clipping_norm)
        self.soft_update_of_target_network(
            self.critic1, 
            self.target_critic1,
            self.tau)
        self.soft_update_of_target_network(
            self.critic2, 
            self.target_critic2,
            self.tau)
                                           
    def update_actor_parameters(self, actor_loss, alpha_loss):
        
        self.take_optimisation_step(
            self.actor_optimizer, 
            self.actor, actor_loss,
            self.gradient_clipping_norm)
        
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """_summary_

        Args:
            local_model (_type_): _description_
            target_model (_type_): _description_
            tau (_type_): _description_
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



