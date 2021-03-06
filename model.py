# Adapted from the autoregressive and parametric actions models detailed here:
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/autoregressive_action_model.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/parametric_actions_model.py

import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

DEFAULT_CONFIG = {
    'seq_len' : 15,
    'lstm_state_size' : 256,
    'id_dim' : 5,
    'id_dim_out' : 5,
    'est_dim' : 6,
    'est_dim_out' : 6,
    'price_dim' : 6,
    'obs_dim_out' : 256,
    'cash_dim' : 1,
    'n_shares_dim' : 1000,
    'a1_dim' : 3,
    'a2_dim' : 1000,
    'context_dim_out' : 128,
    'a2_hidden_size' : 256,
    'action_embedding_dim' : 100
}

class AutoregressiveParametricTradingModel(RecurrentNetwork, nn.Module):
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
        
        RecurrentNetwork.__init__(self, 
                              obs_space, 
                              action_space, 
                              num_outputs,
                              model_config, 
                              name
                             )
        nn.Module.__init__(self)

        for key in layer_config:
            setattr(self, key, layer_config[key])        
        
        self.id_fc = SlimFC(
            in_size=self.id_dim,
            out_size=self.id_dim_out,
            initializer=normc_init_torch(0.01),
            activation_fn=None
        )

        self.est_fc = SlimFC(
            in_size=self.est_dim,
            out_size=self.est_dim_out,
            initializer=normc_init_torch(0.01),
            activation_fn=None
        )

        self.obs_fc = SlimFC(
            in_size=\
                self.price_dim
                + self.cash_dim
                + self.n_shares_dim
                + self.a1_dim
                + self.a2_dim,
            out_size=self.obs_dim_out,
            initializer=normc_init_torch(0.01),
            activation_fn=nn.Tanh
        )

        self.obs_lstm = nn.LSTM(
            self.obs_dim_out,
            self.lstm_state_size,
            batch_first=True
        )        
        
        self.context_layer = SlimFC(
            in_size=\
                self.lstm_state_size
                + self.id_dim_out
                + self.est_dim_out,
            out_size=self.context_dim_out,
            initializer=normc_init_torch(1.0),
            activation_fn=nn.Tanh,
        )   
        
        self.value_branch = SlimFC(
            in_size=self.context_dim_out,
            out_size=1,
            initializer=None,
            activation_fn=None,
        )

        self.a1_logits = SlimFC(
            in_size=self.context_dim_out,
            out_size=3,
            initializer=normc_init_torch(0.01),
            activation_fn=None
        )
        
        class _ActionModel(nn.Module):
            """
            Action distributions for action 1 (buy/sell/hold) and action 2 (amount).
            """
            
            def __init__(self, 
                         context_dim_out, 
                         a2_hidden_size,
                         action_embedding_dim):
                nn.Module.__init__(self)

                self.action_type_mask = None
                self.action_mask = None
                self.action_embeddings = None
                

                self.a2_hidden = SlimFC(
                    in_size=context_dim_out+1,
                    out_size=a2_hidden_size,
                    initializer=normc_init_torch(1.0),
                    activation_fn=nn.Tanh
                )                
                
                self.a2_embedding = SlimFC(
                    in_size=a2_hidden_size,
                    out_size=action_embedding_dim,
                    initializer=normc_init_torch(0.01),
                    activation_fn=None
                )

            def forward(self_, ctx_input, a1_vec):
                """
                Action distribution forward pass. Upsamples action 2 
                embeddings to the MAX_SHARES_TO_SELL and masks
                out invalid actions.
                """

                # a1 (action_type)
                a1_logits = self.a1_logits(ctx_input)

                action_type_mask = torch.clamp(
                    torch.log(self_.action_type_mask), FLOAT_MIN, FLOAT_MAX)
                a1_logits = a1_logits + action_type_mask
                
                # a2 (amount)
                cat_values = torch.cat([ctx_input, a1_vec], dim=1)
                a1_vec = a1_vec.reshape(
                    a1_vec.size(0)).type(torch.LongTensor)
                a2_embedded = self_.a2_embedding(self_.a2_hidden(cat_values))
                action_embeddings = self_.action_embeddings[
                    torch.arange(self_.action_embeddings.size(0)), a1_vec]
                action_mask = self_.action_mask[
                    torch.arange(self_.action_mask.size(0)), a1_vec]
                action_mask = torch.clamp(
                    torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
                a2_logits = torch.sum(
                    a2_embedded.unsqueeze(1) * action_embeddings, dim=2)
                a2_logits = a2_logits + action_mask

                return a1_logits, a2_logits
            
        self.action_module = _ActionModel(
            self.context_dim_out,
            self.a2_hidden_size,
            self.action_embedding_dim)
        self._context = None
    
    @override(RecurrentNetwork)
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

        # ray periodically runs test loops of batch size 1;
        # since the LSTM built here is customized, the state
        # dimensions are off within this loop
        # this ensures that the state is always the right shape
        if state[0].size(0) != input_dict['obs']['price'].size(0):
            state = [
                state[0].new(
                    input_dict['obs']['price'].size(0),
                    self.lstm_state_size
                ).zero_(),
                state[1].new(
                    input_dict['obs']['price'].size(0), 
                    self.lstm_state_size
                ).zero_()
            ]

        lstm_encoding, new_state = self.forward_rnn(
            torch.cat([
                input_dict['obs']['price'],
                input_dict['obs']['cash_record'],
                input_dict['obs']['n_shares_record'],
                input_dict['obs']['a1_record'],
                input_dict['obs']['a2_record']                
            ], dim=2),
            state,
            seq_lens
        )
        
        self._context = self.context_layer(
            torch.cat([
                lstm_encoding[:,-1],
                self.id_fc(input_dict['obs']['symbol_id']),
                self.est_fc(input_dict['obs']['estimate']),
            ], dim=1)        
        )
        
        return self._context, new_state

    @override(RecurrentNetwork)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            self.obs_fc._modules['_model'][0].weight.new(
            1, self.lstm_state_size).zero_().squeeze(0),
            self.obs_fc._modules['_model'][0].weight.new(
            1, self.lstm_state_size).zero_().squeeze(0)
            ]
        return h
        
    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        Base model forward pass. Returns the context of the current state, 
        which will be passed to the action distribution.
        """
        
        x = self.obs_fc(inputs)
        output, [h, c] = self.obs_lstm(
            x, (state[0].unsqueeze(0), 
                state[1].unsqueeze(0)))
        
        return output, [h.squeeze(0), c.squeeze(0)]
    
    def value_function(self):
        """Return the value for the current state (context)."""
        return torch.reshape(self.value_branch(self._context), [-1])