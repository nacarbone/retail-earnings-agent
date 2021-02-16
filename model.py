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
                 fc_size=64,
                 lstm_state_size=256
                ):

        TorchModelV2.__init__(self, 
                              obs_space, 
                              action_space, 
                              num_outputs,
                              model_config, 
                              name
                             )
        nn.Module.__init__(self)
        
        self.obs_fc = SlimFC(
            in_size=6,
            out_size=6,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh
        )

        self.est_fc = SlimFC(
            in_size=7,
            out_size=7,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh
        )

        # update this if we update n_shares to Box space
        self.position_fc = SlimFC(
            in_size=10001,
            out_size=2,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh
        )        

        self.context_layer = SlimFC(
            in_size=6+7+2,
            out_size=256,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh,
        )   
        
        self.value_branch = SlimFC(
            in_size=256,
            out_size=1,
            initializer=normc_init_torch(0.01),
            activation_fn=None,
        )
        
        class _ActionModel(nn.Module):
            """
            Action distributions for action 1 (buy/sell/hold) and action 2 (amount).
            """
            
            def __init__(self):
                nn.Module.__init__(self)

                self.action_type_mask = None
                self.action_mask = None
                self.action_embeddings = None
                
                self.a1_logits = SlimFC(
                    in_size=256+1,
                    out_size=3,
                    initializer=None,
                    activation_fn=nn.Tanh
                )

                self.a2_embedding = SlimFC(
                    in_size=256+1,
                    out_size=100, # update this based on changes to the embedding dimension
                    initializer=None,
                    activation_fn=None
                )

            def forward(self, ctx_input, a1_vec):
                """
                Action distribution forward pass. Upsamples action 2 
                embeddings to the MAX_SHARES_TO_SELL and masks
                out invalid actions.
                """                
#                a1_vec = torch.clamp(a1_vec, 0, 2)
                
                cat_values = torch.cat([ctx_input, a1_vec], dim=1)

                # a1 (action_type)
                a1_logits = self.a1_logits(cat_values)
                action_type_mask = torch.clamp(
                    torch.log(self.action_type_mask), FLOAT_MIN, FLOAT_MAX)
                a1_logits = a1_logits + action_type_mask
                
                # a2 (amount)
                a2_embedded = self.a2_embedding(cat_values)

                action_embeddings = torch.stack([
                    self.action_embeddings[i,int(j.item())] 
                    for i, j in enumerate(a1_vec)
                ])

                action_mask = torch.stack([
                    self.action_mask[i,int(j.item())] 
                    for i, j in enumerate(a1_vec)
                ])
                action_mask = torch.clamp(
                    torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

                a2_logits = torch.sum(
                    a2_embedded.unsqueeze(1) * action_embeddings, dim=2)
                a2_logits = a2_logits + action_mask

                return a1_logits, a2_logits
            
        self.action_module = _ActionModel()
        self._context = None
        
    def forward(self, input_dict, state, seq_lens):
        """
        Base model forward pass. Returns the context of the current state, 
        which will be passed to the action distribution.
        """
        # get rid of these later to avoid copying unnecessarily
        estimate = input_dict['obs']['estimate']
        action_embeddings = input_dict['obs']['action_embeddings']
        cash_balance = input_dict['obs']['cash_balance']
        n_shares = input_dict['obs']['n_shares']
        obs = input_dict['obs']['real_obs']
        
        position = torch.cat([cash_balance, n_shares], dim=1)
        
        self.action_module.action_type_mask = \
            input_dict['obs']['action_type_mask']
        self.action_module.action_mask = \
            input_dict['obs']['action_mask']
        self.action_module.action_embeddings = \
            input_dict['obs']['action_embeddings']
        
        obs_fc_out = self.obs_fc(obs)
        est_fc_out = self.est_fc(estimate)
        position_fc_out = self.position_fc(position)

        cat_values = torch.cat([obs_fc_out, est_fc_out, position_fc_out], dim=1)
        
        self._context = self.context_layer(cat_values)
        
        return self._context, state

    def value_function(self):
        """Return the value for the current state (context)."""
        return torch.reshape(self.value_branch(self._context), [-1])