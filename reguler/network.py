

import torch.nn as nn
import torch
from reguler.helper.embedder import PositionEncoder

class Config:
    pos_in_channels : int = 3
    kp_in_channels: int = 64
    pos_N_freq: int = 10
    pos_max_freq: int = 9
    fusion_channles: int = 256
    post_embed_channels: int = 128
    
    

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
        scale, shift = torch.chunk(emb, 2, dim=0)
        x = self.kp_encoder(x)
        x = x * (1 + scale) + shift   
        return x

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, output_dim)
        self.layer_2 = nn.Linear(output_dim, output_dim)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        output_1 = self.layer_1(x)
        return self.layer_2(output_1)    

class ShiftEstimator(nn.Module):
    def __init__(self, intput_channel: int):
        super(ShiftEstimator, self).__init__()
        self.layer = nn.Linear(intput_channel, 3)
    
    def forward(self, x: torch.Tensor):
        return self.layer(x)    

class Refiner(nn.Module):
    def __init__(self, config: Config):
        super(Refiner, self).__init__()
        self.embed = PositionEncoder(config.pos_in_channels, config.pos_N_freq, config.pos_max_freq)
        embed_output_dim = self.embed.out_dim 
        self.post_embed = MLP(embed_output_dim, config.post_embed_channels)
        self.modulation = Modulation(config.fusion_channles, config.post_embed_channels, config.kp_in_channels)
        self.shift_estimator = ShiftEstimator(config.fusion_channles)
    
    def forward(self, kp_feature: torch.Tensor, pos: torch.Tensor) ->float:
        # position embedding
        pos_embed = self.embed(pos)
        #post processing position embedding dim from 60 to 128
        pos_embed = self.post_embed(pos_embed)
        
        #fusion the feature map
        token = self.modulation(kp_feature, pos_embed)
        
        #estimate shift
        shift = self.shift_estimator(token)
        return shift