
import torch.nn as nn
import torch
from typing import Optional
from .mlp import MLP
from .transformer import LOFTRDecoder, LOFTREncoder

class Config:
    kp_dim: int = 64
    fusion_channles: int = 512
    embed_dim: int = 256
    encode_layer : int = 12
    decoder_layer : int = 8


class PosExtractNet(nn.Module):
    """
    Extract Position (x,y,z) Net
    
    """
    def __init__(self, config: Config):
        super(PosExtractNet, self).__init__()
        """
        Project xfeat from dim = 64 to dim = 256 
        input: dim = 64
        output: dim = 256
        """
        self.proj = nn.Linear(config.kp_dim, config.embed_dim) # dim 64 --> 256

        """
        Reference to "LOFTR"  and "Dustr3R"
        Using LOFTR implementation but the number of layer for encoder matching Dustr3D (encoder 12 layers)
        input : dim = 256
        output: dim = 256
        header = 4
        """

        self.kp_encoder = LOFTREncoder(config.encode_layer)
        self.query_encoder = LOFTREncoder(config.encode_layer)
        
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
        
    
    def forward(self, kp_feature: torch.Tensor, query_feature: torch.Tensor) ->float:
        
        
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



