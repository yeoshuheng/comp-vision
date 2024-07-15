import torch.nn as nn
import torch, math
from einops import rearrange
from utils import unsqueeze_as

### pos embedding ###

class SinusoidalPositionEmb(nn.Module):
    def __init__(self, 
                dim : int,
                max_length : int=10000):
        super().__init__()

        emb = torch.zeros(max_length, dim)
        pos = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(max_length / 2 / math.pi) / dim))

        emb[:, 0::2] = torch.sin(pos * div_term)
        emb[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('embedding', emb)

    def forward(self, x):
        # discrete x [BS,]
        return self.embedding[x]
    
class EmbeddingFFN(nn.Module):
    def __init__(self, 
                in_dim : int, 
                embed_dim : int):
        super().__init__()
        self.init_embed = nn.Linear(in_dim, embed_dim)
        self.time_embed = SinusoidalPositionEmb(embed_dim)
        self.model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, in_dim),
        )

    def forward(self, x, t):
        x = self.init_embed(x)
        t = self.time_embed(t)
        return self.model(x + t)

### conv blocks ###

class BasicConvBlock(nn.Module):
    def __init__(self, 
                in_channel : int, 
                out_channel : int, 
                time_channel : int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.mlp_time = nn.Sequential(
            nn.Linear(time_channel, time_channel),
            nn.ReLU(),
            nn.Linear(time_channel, out_channel),
        )
        self.resid_align = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel)
            ) if in_channel != out_channel else nn.Identity()
        
        self.relu = nn.ReLU()

    def forward(self, x, t):

        out = self.conv1(x)
        out = self.bn1(out)

        # add time embedding
        out += unsqueeze_as(self.mlp_time(t), x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # residual jump
        out = self.relu(out + self.resid_align(x))
        return out
    
class SelfAttention2dBlock(nn.Module):
    def __init__(self,
                 dims : int,
                 num_heads : int=8,
                 dropout : float=0.1):
        super().__init__()

        assert num_heads % dims == 0 # head should be divisible by d!

        self.d = dims
        self.num_heads = num_heads
        self.d_k = num_heads // dims

        # k, q, v, o weights
        self.W_k = nn.Conv2d(dims, dims, 1, bias=True)
        self.W_q = nn.Conv2d(dims, dims, 1, bias=True)
        self.W_v = nn.Conv2d(dims, dims, 1, bias=True)
        self.W_o = nn.Conv2d(dims, dims, 1, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # get k, q, v
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)

        # rearrange to groups for multi-head attention
        # [BS, (N_H * NUM_CHANNEL_PER_HEAD (N_C_H)), H, W] =>
        # [(BS * N_H), N_C_H, (H * W)]
        k = rearrange(k, "b (g c) h w -> (b g) c (h w)", g = self.num_heads)
        q = rearrange(q, "b (g c) h w -> (b g) c (h w)", g = self.num_heads)
        v = rearrange(v, "b (g c) h w -> (b g) c (h w)", g = self.num_heads)

        # get attention (dot pdt across head / batch)
        # [(BS * N_H), N_C_H, (H * W)] & [(BS * N_H), N_C_H, (H * W)]  => 
        # [(BS * N_H), (H * W), (H * W)]

        attn = torch.einsum("b c s, b c t -> b s t", q, k) / self.d ** 0.5
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

         # [(BS * N_H), (H * W), (H * W)] & [(BS * N_H), N_C_H, (H * W)]  => 
        # [(BS * N_H), (N_C_H), (H * W)]
        o = torch.einsum("b s t, b c t -> b c s", attn, v)

        # remove head
        o = rearrange(o, "(b g) s (h w) -> b (g s) h w", g = self.num_heads, w = x.shape[1])
        o = self.W_o(o)
        
        return x + o

