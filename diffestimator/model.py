
import torch.nn as nn
import torch
from typing import Optional
from .mlp import MLP
from .transformer import LOFTRDecoder, LOFTREncoder

class Config:
    kp_dim: int = 64
    fusion_channles: int = 512
    embed_dim: int = 256
    encoder_layer : int = 12
    decoder_layer : int = 8
    pose_dim : int = 12 # (3x4 = 3x3 (R) + 3x1 (t))


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
        Only self attention
        input : dim = 256
        output: dim = 256
        """

        self.kp_encoder = LOFTREncoder(config.encoder_layer)
        self.query_encoder = LOFTREncoder(config.encoder_layer)
        
        """
        Reference to "LOFTR"  and "Dustr3R"
        Using LOFTR implementation but the number of layer for decoder matching Dustr3D (encoder 8 layers)
        Only cross attention
        input : dim = 256
        output: dim = 256
        """
        self.decoder = LOFTRDecoder(config.decoder_layer)
     
        """
        Regress the final pose 3x4
        1 x linear , 1 x activation (relu), 1x Linear 
        input : dim = 256
        hidden : dim = 1024  (projet to higher dimension)
        output : dim = 3
        """
        self.estim_pose = MLP(config.fusion_channles, 12,1024, 1, "relu")
        
    
    def forward(self, kp_feature: torch.Tensor, query_feature: torch.Tensor) ->float:
        
        # project keypoint matched feature and query feature to higher dimension  
        kp_feature = self.proj(kp_feature)
        query_feature = self.proj(query_feature)
 
        #Encode the feature
        kp_encode = self.kp_encoder(kp_feature)
        query_encode = self.query_encoder(query_feature)

        #Decode the feature
        kp_decode, query_decode = self.decoder(kp_encode, query_encode)
        
        # catenate position encoding and xfeat feature
        kp_query_fea = torch.cat([query_decode, kp_decode], dim=2)
        
        
        #estimate shift
        predict_pose = self.estim_pose(kp_query_fea)
        predict_pose = predict_pose.reshape((kp_query_fea.size(0), 3,4))
        predict_R = predict_pose[:,:3,:3]
        predict_t = predict_pose[:,:3,3]
        
        return predict_R, predict_t



