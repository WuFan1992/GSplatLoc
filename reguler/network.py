

import torch.nn as nn
import torch
from reguler.helper.embedder import PositionEncoder
from reguler.helper.utils import get_activation
from typing import Optional

class Config:
    pos_in_channels : int = 3
    kp_in_channels: int = 64
    pos_N_freq: int = 10
    pos_max_freq: int = 9
    fusion_channles: int = 256
    post_embed_channels: int = 128
    density_channel : int = 1
    mass_center_channel : int = 3
    
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
    
    

class Modulation(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int, kp_feature_dim: int, single_layer: bool = False):
        super().__init__()
        self.silu = nn.SiLU()
        if single_layer:
            self.linear1 = nn.Identity()
        else:
            self.linear1 = nn.Linear(condition_dim, condition_dim)

        self.linear2 = nn.Linear(condition_dim, embedding_dim * 2)
        self.kp_encoder = nn.Linear(kp_feature_dim, embedding_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        emb = self.linear2(self.silu(self.linear1(condition)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.kp_encoder(x)
        x = x * (1 + scale) + shift   
        return x

"""
 A standard MLP layer 
 
"""
class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(output_activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        elif activation == "LeakyReLU":
            return nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError



"""
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.layer_1 = nn.Linear(input_dim, output_dim)
        self.layer_2 = nn.Linear(output_dim, output_dim)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        output_1 = self.activation(self.layer_1(x))
        return self.activation(self.layer_2(output_1))    


"""
class ShiftEstimator(nn.Module):
    def __init__(self, intput_channel: int):
        super(ShiftEstimator, self).__init__()
        self.activation = nn.Tanh()
        self.layer = nn.Linear(intput_channel, 3)
    
    def forward(self, x: torch.Tensor):
        return self.activation(self.layer(x))    

class Refiner(nn.Module):
    def __init__(self, config: Config):
        super(Refiner, self).__init__()
        self.embed = PositionEncoder(config.pos_in_channels, config.pos_N_freq, config.pos_max_freq)
        embed_output_dim = self.embed.out_dim 
        self.post_embed = MLP(embed_output_dim, config.post_embed_channels, config.post_embed_channels, 2, "LeakyReLU", output_activation="relu")
        self.pre_kp_features = MLP(config.kp_in_channels, config.kp_in_channels, config.kp_in_channels, 4,"LeakyReLU", output_activation="relu")
        self.process_density = MLP(config.density_channel+config.mass_center_channel,  config.post_embed_channels, config.post_embed_channels, 1, "silu")
        #self.post_embed.apply(weights_init_uniform)
        self.modulation = Modulation(config.fusion_channles, config.post_embed_channels, config.kp_in_channels)
        #self.modulation.apply(weights_init_uniform)
        self.shift_estimator = ShiftEstimator(config.fusion_channles)
        self.feature_updator = MLP(config.fusion_channles, config.kp_in_channels, config. kp_in_channels, 4,"LeakyReLU", output_activation="relu")
        #self.shift_estimator.apply(weights_init_uniform)
    
    def forward(self, kp_feature: torch.Tensor, mass_den: torch.Tensor) ->float:
        
        # position embedding

        #pos_embed = self.embed(pos)
 
        #post processing position embedding dim from 60 to 128
        #pos_embed = self.post_embed(pos_embed)
        mass_den_embed = self.process_density(mass_den)
        
        #fusion the feature map
        kp_feature = self.pre_kp_features(kp_feature)
        token = self.modulation(kp_feature, mass_den_embed)

        #estimate shift
        shift = self.shift_estimator(token)
        feature = self.feature_updator(token)

        return shift, feature