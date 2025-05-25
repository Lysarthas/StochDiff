import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class SelfAttention(nn.Module):
    def __init__(self, x_dim, heads = 4, dim_head = 64, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(x_dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, x_dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)
    
class CrossAttention(nn.Module):
    def __init__(self, x_dim, cond_dim, num_heads, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * num_heads

        self.scale = dim_head ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(x_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, x_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond):
        h = self.heads
        # print(x.shape, cond.shape)
        q = self.to_q(x).unsqueeze(-1)
        k = self.to_k(cond).unsqueeze(-1)
        v = self.to_v(cond).unsqueeze(-1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h c n -> b (h c n)', h = self.heads)
        return self.to_out(out)

class FeatureAttention(nn.Module):
    def __init__(self, x_dim, out_dim, num_heads, dim_head, dropout=0.):
        super(FeatureAttention, self).__init__()
#         # self.token = nn.Parameter(torch.randn(1, tk_dim))
#         # self.atten_tk = CrossAttention(tk_dim, x_dim, num_heads, dropout=dropout)
        self.atten_x = SelfAttention(x_dim, num_heads, dim_head, dropout=dropout)
#         # self.ffd_tk = nn.Linear(tk_dim, tk_dim)
#         # self.norm_tk = nn.LayerNorm(tk_dim)
        self.ffd_x = nn.Linear(x_dim, x_dim)
        self.norm_x = nn.LayerNorm(x_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(x_dim, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch_size, in_dim, 1]
        out = self.atten_x(x.unsqueeze(-1))
        out = self.norm_x(out.squeeze(-1) + x)
        out = self.out_proj(out)
        
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, diff_dim, cond_dim, res_dim, num_heads, dropout=0):
        super(ResidualBlock, self).__init__()

        self.diffproj = nn.Linear(diff_dim, 2*res_dim)
        self.condproj = nn.Linear(cond_dim, 2*res_dim)
        self.selfattention = SelfAttention(in_dim, num_heads, dropout=dropout)
        self.condattention = CrossAttention(in_dim, 2*res_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_dim)
        # self.norm2 = nn.LayerNorm(in_dim)
        self.outproj = nn.Linear(res_dim, 2*in_dim)

    def forward(self, x, conds, diff_emb):
        # x: [batch_size, in_dim]
        res = x
        diff_proj = self.diffproj(diff_emb)
        # print(x.shape, diff_proj.shape)
        x = x + diff_proj
        x = self.selfattention(x.unsqueeze(-1)).squeeze(-1)
        x = self.norm1(x)
        cond = self.condproj(conds)
        x = self.condattention(x, cond)
        # x = self.norm2(x)
        
        gate, filt = torch.chunk(x, 2, dim=1)
        
        x = torch.tanh(gate) * torch.sigmoid(filt)
        
        x = self.outproj(x)
        x = F.leaky_relu(x, 0.4)
        out, skip = torch.chunk(x, 2, dim=1)
        out = (res + out) / torch.sqrt(torch.tensor(2.0))

        return out, skip
    
class DiffusionEmbedding(nn.Module):

    def __init__(self, emb_dim, h_dim, max_steps=500):
        super(DiffusionEmbedding, self).__init__()

        self.register_buffer(
            "embedding", self.get_diff_step_embedding(emb_dim, max_steps), persistent=False
        )
        self.diff_embed = nn.Sequential(
            nn.Linear(2*emb_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU()
        )

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.diff_embed(x)
        return x

    def get_diff_step_embedding(self, diff_step_embed_dim, max_steps):

        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(diff_step_embed_dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / diff_step_embed_dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class AttentionNet(nn.Module):

    def __init__(self, x_dim, h_dim, cond_dim, diff_step_emb_dim, num_heads, dropout=0):
        super(AttentionNet, self).__init__()

        self.diff_emb = DiffusionEmbedding(diff_step_emb_dim, h_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.LeakyReLU()
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_dim=h_dim,
                    diff_dim=h_dim,
                    cond_dim=cond_dim,
                    res_dim=h_dim//2,
                    num_heads=num_heads,
                    dropout=dropout
                )
            ]
        )

        self.skip_proj = nn.Linear(h_dim, h_dim)
        self.out_proj = nn.Linear(h_dim, x_dim)

    def forward(self, x, diff_t, conds):
        # x: [batch_size, seq_len, in_dim]
        # conds: [batch_size, cond_dim]
        # diff_step: [batch_size, diff_step_emb_dim]
        x = self.input_proj(x)
        diff_emb = self.diff_emb(diff_t)
        skip = []
        for layer in self.residual_layers:
            x, s = layer(x, conds, diff_emb)
            skip.append(s)
        
        x = torch.sum(torch.stack(skip), dim=0) / torch.sqrt(torch.tensor(len(self.residual_layers)))
        x = self.skip_proj(x)
        x = F.leaky_relu(x, 0.4)
        x = self.out_proj(x)

        return x
    
# class TemporalConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
#         super(TemporalConv, self).__init__()
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        
#     def forward(self, x):
#         padding = (self.kernel_size - 1) * self.dilation
#         x = F.pad(x, (padding, 0))
#         x = self.conv1d(x)
#         return x
    
# class TCN_ResBlock(nn.Module):
#     def __init__(self, num_residual_channels, num_skip_channels, dilation_rate):
#         super(TCN_ResBlock, self).__init__()
#         self.temp_conv = TemporalConv(num_residual_channels, num_residual_channels, kernel_size=2, dilation=dilation_rate)
#         self.residual_conv = nn.Conv1d(num_residual_channels, num_residual_channels, kernel_size=1)
#         self.skip_conv = nn.Conv1d(num_residual_channels, num_skip_channels, kernel_size=1)

#     def forward(self, x):
#         temp_output = self.temp_conv(x)
#         residual_output = self.residual_conv(temp_output)
#         skip_output = self.skip_conv(temp_output)
#         output = x + residual_output
#         return output, skip_output

# class TCN(nn.Module):
#     def __init__(self, in_dim, out_dim, num_blocks, num_residual_channels, num_skip_channels, dilation_rates=[1,2,4]):
#         super(TCN, self).__init__()
#         self.num_blocks = num_blocks
#         self.num_residual_channels = num_residual_channels
#         self.num_skip_channels = num_skip_channels
#         self.dilation_rates = dilation_rates

#         self.input_conv = TemporalConv(in_dim, num_residual_channels, kernel_size=1)

#         self.residual_layers = nn.ModuleList()

#         for _ in range(num_blocks):
#             for dilation_rate in dilation_rates:
#                 self.residual_layers.append(TCN_ResBlock(num_residual_channels, num_skip_channels, dilation_rate))

#         # self.output_conv1 = nn.Conv1d(num_skip_channels, num_skip_channels, kernel_size=1)
#         self.output_conv = nn.Conv1d(num_skip_channels, in_dim, kernel_size=1)
#         self.output_proj = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         x = self.input_conv(x)

#         skip_connections = []
#         for layer in self.residual_layers:
#             x, skip = layer(x)
#             skip_connections.append(skip)

#         x = torch.sum(torch.stack(skip_connections), dim=0) / torch.sqrt(torch.tensor(len(self.residual_layers)))
#         x = F.relu(x)
#         x = self.output_conv(x)
#         x = F.relu(x)
#         x = self.output_proj(x[:, :, -1])

#         return x