# Adapted from the autoregressive and parametric actions models detailed here:
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/autoregressive_action_model.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/parametric_actions_model.py

import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

default_config = {
    'obs_dim' : 6,
    'est_dim' : 7,
    'position_dim' : 10000 + 1,
    'position_dim_out' : 2,
    'context_dim_out' : 256,
    'action_embedding_dim' : 100
}

class AutoregressiveParametricTradingModel(TorchModelV2, nn.Module):
    """
    Custom model for a parametric actions space with action 2 (amount)
    conditioned on action 1 (buy/sell/hold).
    """
    
    
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **layer_config
                ):
        
        TorchModelV2.__init__(self, 
                              obs_space, 
                              action_space, 
                              num_outputs,
                              model_config, 
                              name
                             )
        nn.Module.__init__(self)
        
        for key in layer_config:
            setattr(self, key, layer_config[key])        
        
        self.obs_fc = SlimFC(
            in_size=self.obs_dim,
            out_size=self.obs_dim,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh
        )

        self.est_fc = SlimFC(
            in_size=self.est_dim,
            out_size=self.est_dim,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh
        )

        self.position_fc = SlimFC(
            in_size=self.position_dim + 1,
            out_size=self.position_dim_out,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh
        )

        self.context_layer = SlimFC(
            in_size=\
                self.obs_dim
                + self.est_dim
                + self.position_dim_out,
            out_size=self.context_dim_out,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh,
        )   
        
        self.value_branch = SlimFC(
            in_size=self.context_dim_out,
            out_size=1,
            initializer=normc_init_torch(0.01),
            activation_fn=None,
        )
        
        class _ActionModel(nn.Module):
            """
            Action distributions for action 1 (buy/sell/hold) and action 2 (amount).
            """
            
            def __init__(self, context_dim_out, action_embedding_dim):
                nn.Module.__init__(self)

                self.action_type_mask = None
                self.action_mask = None
                self.action_embeddings = None
                
                self.a1_logits = SlimFC(
                    in_size=context_dim_out+1,
                    out_size=3,
                    initializer=None,
                    activation_fn=nn.Tanh
                )

                self.a2_embedding = SlimFC(
                    in_size=context_dim_out+1,
                    out_size=action_embedding_dim,
                    initializer=None,
                    activation_fn=nn.Tanh
                )

            def forward(self, ctx_input, a1_vec):
                """
                Action distribution forward pass. Upsamples action 2 
                embeddings to the MAX_SHARES_TO_SELL and masks
                out invalid actions.
                """
                cat_values = torch.cat([ctx_input, a1_vec], dim=1)

                # a1 (action_type)
                a1_logits = self.a1_logits(cat_values)
                action_type_mask = torch.clamp(
                    torch.log(self.action_type_mask), FLOAT_MIN, FLOAT_MAX)
                a1_logits = a1_logits + action_type_mask
                
                # a2 (amount)
                a1_vec = a1_vec.reshape(
                    a1_vec.size(0)).type(torch.LongTensor)
                a2_embedded = self.a2_embedding(cat_values)
                action_embeddings = self.action_embeddings[
                    torch.arange(self.action_embeddings.size(0)), a1_vec]
                action_mask = self.action_mask[
                    torch.arange(self.action_mask.size(0)), a1_vec]
                action_mask = torch.clamp(
                    torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
                a2_logits = torch.sum(
                    a2_embedded.unsqueeze(1) * action_embeddings, dim=2)
                a2_logits = a2_logits + action_mask

                return a1_logits, a2_logits
            
        self.action_module = _ActionModel(
            self.context_dim_out,
            self.action_embedding_dim)
        self._context = None
        
    def forward(self, input_dict, state, seq_lens):
        """
        Base model forward pass. Returns the context of the current state, 
        which will be passed to the action distribution.
        """
        self.action_module.action_type_mask = \
            input_dict['obs']['action_type_mask']
        self.action_module.action_mask = \
            input_dict['obs']['action_mask']
        self.action_module.action_embeddings = \
            input_dict['obs']['action_embeddings']

        self._context = self.context_layer(
            torch.cat([
                self.obs_fc(input_dict['obs']['real_obs']),
                self.est_fc(input_dict['obs']['estimate']),
                self.position_fc(torch.cat([
                    input_dict['obs']['cash_balance'],
                    input_dict['obs']['n_shares']
                ], dim=1))
            ], dim=1)        
        )
        
        return self._context, state

    def value_function(self):
        """Return the value for the current state (context)."""
        return torch.reshape(self.value_branch(self._context), [-1])