import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from truncated_normal import TruncatedNormalDist
from .networks import DiscretePolicyNetwork, QNetworkNoAction
from .SAC import SAC

class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"
    def __init__(
            self, 
            state_dim, 
            discrete_action_dim,
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
        self.discrete_action_dim = discrete_action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.device=device
        self.gaussian_lower_bound = gaussian_lower_bound
        self.gaussian_upper_bound = gaussian_upper_bound

        self.EPSILON = 1e-6
        
        self.actor = DiscretePolicyNetwork(state_dim, discrete_action_dim, hidden_size)
        self.critic1 = QNetworkNoAction(state_dim, discrete_action_dim, hidden_size)
        self.critic2 = QNetworkNoAction(state_dim, discrete_action_dim, hidden_size)
        self.target_critic1 = QNetworkNoAction(state_dim, discrete_action_dim, hidden_size)
        self.target_critic2 = QNetworkNoAction(state_dim, discrete_action_dim, hidden_size)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = torch.optim.Adam(list(self.critic1.parameters()), lr=self.lr)
        self.critic2_optimizer = torch.optim.Adam(list(self.critic2.parameters()), lr=self.lr)


        self.automatic_entropy_tuning = auto_tune_entropy
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.discrete_action_dim).to(self.device)).item() # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)

    def produce_action_and_action_info(self, state):
        """_summary_

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        action_probabilities = self.actor(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = torch.distributions.Categorical(action_probabilities)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

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
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.target_critic1(next_state_batch)
            qf2_next_target = self.target_critic2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch.unsqueeze(-1) + (1.0 - mask_batch.unsqueeze(-1)) * self.gamma * (min_qf_next_target)


        qf1 = self.critic1(state_batch).gather(1, action_batch.long())
        qf2 = self.critic2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic1(state_batch)
        qf2_pi = self.critic2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities
