

import torch.nn as nn
import torch
from reguler.helper.embedder import PositionEncoder
from reguler.helper.utils import get_activation
from typing import Optional
from reguler.helper.transformer import LoFTREncoderLayer

class Config:
    pos_in_channels : int = 3
    kp_in_channels: int = 64
    pos_N_freq: int = 10
    pos_max_freq: int = 9
    fusion_channles: int = 512
    post_embed_channels: int = 256
    density_channel : int = 1
    mass_center_channel : int = 3
    camera_embed_input_channel: int = 25
    transformer_header_num : int = 4
    
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
    
    



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
    
    
class FeatureEstimator(nn.Module):
    def __init__(self, intput_channel: int):
        super(FeatureEstimator, self).__init__()
        self.activation = nn.Tanh()
        self.layer = nn.Linear(intput_channel, 64)
    
    def forward(self, x: torch.Tensor):
        return self.activation(self.layer(x))   

class Refiner(nn.Module):
    def __init__(self, config: Config):
        super(Refiner, self).__init__()
        """
        Reference to "NeRF" Position Embedding 
        input [x,y,z], dim = 3
        output: dim = 60
        """
        self.embed = PositionEncoder(config.pos_in_channels, config.pos_N_freq, config.pos_max_freq) # (3,10,10)
        embed_output_dim = self.embed.out_dim 
        """
        Reference to "NeRF"  backbone module:
          project position embedding to higher dimension
          1 x Linear, No activation
          input : dim = 60
          output : dim = 256 (The same as position embedding in NeRF and in LOFTR )
        """
        self.post_embed = nn.Linear(embed_output_dim, config.post_embed_channels)
        
        #self.post_embed = MLP(embed_output_dim, config.post_embed_channels, config.post_embed_channels, 1, "silu")
        
        """
        xfeat dim = 64 ---> dim = 256
        
        """
        self.pre_kp_feature = nn.Linear(config.kp_in_channels, config.post_embed_channels)

        """
        Reference to "Triplane meets Gaussian Splatting " camera embedding module:
        project camera extrinsic and intrinsic to higher dimension
         1 x Linear, 1 x activation (silu), 1 x Linear
          input : dim = 25
          output : dim = 256 (The same as position embedding in NeRF and in LOFTR )
        """
        self.camera_embed = MLP(config.camera_embed_input_channel, config.post_embed_channels, config.post_embed_channels, 1, "silu")
        
        """
        Reference to "LOFTR" and "Triplane meets Gaussian"
        4 layers transformer(with modulation)(LOFTR) 
        input : dim = 256
        output: dim = 256
        header = 4
        """
        self.self_atten = LoFTREncoderLayer(config.fusion_channles, config.post_embed_channels, config.transformer_header_num)
        
        """
        Regress the final position
        1 x linear , 1 x activation (relu), 1x Linear 
        input : dim = 256
        hidden : dim = 1024  (projet to higher dimension)
        output : dim = 3
        """
        self.estim_pos = MLP(config.post_embed_channels, 3,1024, 1, "relu")
        
        """
        Regress the final descriptor
        4 x mlp 
        input : dim = 256
        hidden : dim = 1024  (projet to higher dimension)
        output : dim = 64
        """
        self.estim_feature = MLP(config.post_embed_channels, 64,1024, 4, "relu")
        
        
        #self.process_density = MLP(config.density_channel+config.mass_center_channel,  config.post_embed_channels, config.post_embed_channels, 1, "silu")
        #self.post_embed.apply(weights_init_uniform)
        #self.modulation.apply(weights_init_uniform)
        #self.shift_estimator = ShiftEstimator(config.fusion_channles)
        #self.feature_updator = MLP(config.fusion_channles, config.kp_in_channels, config. kp_in_channels, 4,"LeakyReLU")
        #self.feature_updator = FeatureEstimator(config.fusion_channles)
        #self.shift_estimator.apply(weights_init_uniform)
    
    def forward(self, kp_feature: torch.Tensor, pos: torch.Tensor, cam_intr: torch.Tensor, cam_extr: torch.Tensor) ->float:
        
        
        # position embedding
        pos_embed = self.embed(pos)
 
        #post processing position embedding dim from 60 to 256
        pos_embed = self.post_embed(pos_embed)

        #fusion the feature map
        kp_feature = self.pre_kp_feature(kp_feature)
        
        # catenate position encoding and xfeat feature
        kp_pos_encod = torch.cat([pos_embed, kp_feature], dim=1)
        
        # prepare camera embedding input
        cam_data = torch.cat([cam_intr, cam_extr], dim=1)
        cam_data = self.camera_embed(cam_data)
        
        # self attention condition with 
        token = self.self_atten(kp_pos_encod, kp_pos_encod, cam_data)
        
        #estimate shift
        shift = self.estim_pos(token)
        feature = self.estim_feature(token)

        return shift, feature



