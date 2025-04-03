import torch.nn as nn
import torch 


class Modulation(nn.Module):
    """
    Reference to VGGT and Triplane meets Gaussian, originate from DINO V2
    Objectif: fusion feature with the position/camera pose
    How:
        Use Silu activation function to project position embedding/camera pose embedding into higher dimension (x2 in position or x3 in pose)
        Seperate the higher dimension embedding into scale and shift (each one have the same dim as the original embedding)
        fusion feature = feature * (1 + scale) + shift  

    """
    def __init__(self, embedding_dim: int, condition_dim: int, zero_init: bool = False, single_layer: bool = False):
        super().__init__()
        self.silu = nn.SiLU()
        if single_layer:
            self.linear1 = nn.Identity()
        else:
            self.linear1 = nn.Linear(condition_dim, condition_dim)

        self.linear2 = nn.Linear(condition_dim, embedding_dim * 2)
        
        # Only zero init the last linear layer
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        emb = self.linear2(self.silu(self.linear1(condition)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale) + shift   
        return x