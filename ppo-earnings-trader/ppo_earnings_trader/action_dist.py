from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchDistributionWrapper
torch, nn = try_import_torch()

CONTEXT_OUTPUT_SIZE = 128

class TorchMultinomialAutoregressiveDistribution(TorchDistributionWrapper):
    """
    Action distribution P(a1, a2) = P(a1) * P(a2 | a1). For implementation
    details, see:

    https://github.com/ray-project/ray/blob/master/rllib/examples/models/
    autoregressive_action_dist.py
    """

    def deterministic_sample(self):
        """
        Deterministically sample action 1 (buy/sell/hold) and then action 2
        (amount) conditioned action 1.

        Returns
        ---
        A dict of a1 and a2
        """
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        return {'a1' : a1, 'a2' : a2}

    def sample(self):
        """
        Sample action 1 (buy/sell/hold) and then action 2 (amount)
        conditioned action 1.

        Returns
        ---
        A dict of a1 and a2
        """
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        return {'a1' : a1, 'a2' : a2}

    def logp(self, actions: 'torch.LongTensor'):
        """
        Gets the log probabilities of the current action distribution

        Parameters
        ---
        actions : torch.LongTensor
            The actions selected by the model

        Returns
        ---
        The sum log probabilities of the current action distribution.
        """
        # Note on dict action spaces:
        # ray will sort actions alphabetically
        a1, a2 = actions[:,0], actions[:,1]
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

        Returns
        ---
        The sum log probabilities of the current action distribution.        ---
        """
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())

        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other: 'ray.rllib.policy.policy'):
        """
        Calculates the KL divergence between the old and updated action
        distributions.

        Parameters
        ---
        other : ray.rllib.policy.policy
            The old policy to which we're comparing
        """
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))

        return a1_terms + a2_terms

    def _a1_distribution(self):
        """
        Samples action 1 (buy/sell/hold) independently.

        Returns
        ---
        A distribution for action 1 based on the current policy
        """
        BATCH = self.inputs.shape[0]
        zeros = torch.zeros((BATCH, 1)).to(self.inputs.device)
        a1_logits, _ = self.model.action_module(self.inputs, zeros)
        a1_dist = TorchCategorical(a1_logits)

        return a1_dist

    def _a2_distribution(self, a1: 'torch.LongTensor'):
        """
        Samples action 2 (amount) conditioned on action 1 (buy/sell/hold).

        Parameters
        ---
        a1 : torch.LongTensor
            The action 1s taken by the model

        Returns
        ---
        A distribution for action 2 based on the current policy
        """
        a1_vec = torch.unsqueeze(a1.float(), 1)
        _, a2_logits = self.model.action_module(self.inputs, a1_vec)
        a2_dist = TorchCategorical(a2_logits)

        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space: 'gym.Spaces',
                                    model_config: dict):
        """
        Returns the required input shape to the action distribution.

        Parameters
        ---
        action_space : gym.Spaces
            The action space used by the environment
        model_config : dict
            The configuration for the model
        """
        return CONTEXT_OUTPUT_SIZE
