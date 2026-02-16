import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        if self.discrete:
            logits = self.forward(ptu.from_numpy(obs).unsqueeze(0)) # in: (1, obs) out: (1, 1, action_dim)
            action = torch.multinomial(logits, num_samples=1) # out: (1, 1)
            return ptu.to_numpy(action).squeeze() # scalar numpy
        else:
            mean, std = self.forward(ptu.from_numpy(obs).unsqueeze(0)) # in: (1, obs) out: (1, aciton_dim), (action_dim)
            action = torch.distributions.Normal(mean, std).sample()
            return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            return F.softmax(self.logits_net(obs), dim=-1) # (B, H) -> (B, H, L)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            return self.mean_net(obs), torch.exp(self.logstd) # (B, H, L), (L)
    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: compute the policy gradient actor loss
        if self.discrete:
            # actions is B by H
            # self.forward results in (B, H, L)
            
            
            negative_likelihoods = F.cross_entropy(self.forward(obs), actions.long()) # emits (B*H)
            weighted_likelihoods = negative_likelihoods * advantages # (B*H)
            # log_probs = torch.log(torch.gather(self.forward(obs), dim=-1, index=actions)) # results in B H
            # log_probs_times_q = log_probs * advantages # (B, H) * (B, H) elementwise = (B, H)
            # trajectory_wise_sum = torch.sum(log_probs_times_q, dim = -1) # (B, H) -> (B)
            # loss = torch.mean(trajectory_wise_sum) # (B) -> 1
        else:
            # use torch.distributions.Normal and smaple then just call logprob to get logprob
            mean, std_vec = self.forward(obs) # (B * H, L) (L, L)
            action_distributions = torch.distributions.Normal(mean, std_vec) # input: (B*H, L), (L)
            log_probs = action_distributions.log_prob(actions).sum(dim=-1) # (B*H)
            weighted_likelihoods = -log_probs * advantages # (B*H)
            
            
        loss = torch.mean(weighted_likelihoods) # per step loss
        
        

        # TODO: perform an optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
