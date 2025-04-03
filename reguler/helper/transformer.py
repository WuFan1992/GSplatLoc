import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
from .modulation import Modulation


class Transformer_Config:
    dim_model : int = 64
    num_head: int = 2
    attention: str = "linear"
    

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()
        
        # Afrer cat pos and feature, dim = 512, we need to mlp to project it into dim=256
        self.adapdim = nn.Linear(input_dim, d_model)

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # modulation
        """
        Reference to  "Triplane meets Gaussian" Dinov2Layer implementation
        Architecture: 2 modulation model 
        
        """
        self.modulation1 = Modulation(d_model, d_model, zero_init=True, single_layer=True)
        self.modulation2 = Modulation(d_model, d_model, zero_init=True, single_layer=True)

    def forward(self, x, source, modulation_cond, x_mask=None, source_mask=None):
        """
        Reference to "Triplane meets Gaussian " Dinov2Layer implementation, the order of 
        feedforward:
             * normalization feature
             * apply modulation with camera embedding
             * self attention
             * self attention output + original feature -->  (1)
             * normalization (1) ---> (2)
             * apply modulation with (2)
             * mlp  (3)
             * (2) + (3) 
        
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        x, source = self.adapdim(x)[None], self.adapdim(source)[None]
        bs = x.size(0)
        # norm 
        x, source = self.norm0(x), self.norm0(source)
        #modulation
        x = self.modulation1(x, modulation_cond)
        source = self.modulation1(source, modulation_cond)
        # get q k v
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)
        message = self.modulation2(message, modulation_cond)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


    
class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, num_layer):
        super(LocalFeatureTransformer, self).__init__()

        self.config = Transformer_Config()
        self.d_model = self.config.dim_model
        self.num_head = self.config.num_head
        self.attention = self.config.attention
        encoder_layer = LoFTREncoderLayer(self.d_model, self.num_head, self.attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layer)])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, mask=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            mask0 (torch.Tensor): [N, L] (optional)
        """

        assert self.d_model == feat.size(2), "the feature number of src and transformer must be equal"
        for layer in self.layers:
            feat = layer(feat, feat, mask, mask)  
        return feat

