# Follows the paradigm set out here:
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/autoregressive_action_dist.py

import ray
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchDistributionWrapper
torch, nn = try_import_torch()

class TorchMultinomialAutoregressiveDistribution(TorchDistributionWrapper):
    """
    Action distribution P(a1, a2) = P(a1) * P(a2 | a1)
    """

    def deterministic_sample(self):
        """
        Deterministically sample action 1 (buy/sell/hold) and then action 2 
        (amount) conditioned action 1.
        """
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()
        
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)
        
        return {'buy/sell/hold' : a1, 'amount' : a2}

    def sample(self):
        """
        Sample action 1 (buy/sell/hold) and then action 2 (amount) 
        conditioned action 1.
        """
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()        

        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)
        
        return {'buy/sell/hold' : a1, 'amount' : a2}

    def logp(self, actions):
        """
        Return the log probabilities of the current action distribution.
        """
        a1, a2 = actions[:,1], actions[:,0] # the actions are flipped? Ray must sort the actions by name in dict
        a1_vec = torch.unsqueeze(a1.float(), 1)
        a1_logits, a2_logits = self.model.action_module(
            self.inputs, a1_vec)
        
        return (TorchCategorical(a1_logits).logp(a1) +
                    TorchCategorical(a2_logits).logp(a2))
        
    def sampled_action_logp(self):
        return torch.exp(self._action_logp)

    def entropy(self):
        """
        Returns the total entropy of the current action distribuion.
        """
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        """
        Calculates the KL divergence between the old and updated action 
        distributions.
        """
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        
        return a1_terms + a2_terms

    def _a1_distribution(self):
        """
        Samples action 1 (buy/sell/hold) independently.
        """
        BATCH = self.inputs.shape[0]
        zeros = torch.zeros((BATCH, 1)).to(self.inputs.device)
        a1_logits, _ = self.model.action_module(self.inputs, zeros)
        a1_dist = TorchCategorical(a1_logits)
        
        return a1_dist

    def _a2_distribution(self, a1):
        """
        Samples action 2 (amount) conditioned on action 1 (buy/sell/hold).
        """
        a1_vec = torch.unsqueeze(a1.float(), 1)
        _, a2_logits = self.model.action_module(self.inputs, a1_vec)
        a2_dist = TorchCategorical(a2_logits)
        
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """
        Returns the required input shape to the action distribution.
        """
        # this needs to be given more thought and updated
        # first guess is that it should be 256 based on the context layer output
        return 100