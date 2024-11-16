import torch
from torch import nn
import torch.nn.functional as F

from mcm.models.mwnet.film import FiLM


class TimeWiseSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = FiLM(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = self.key(self.norm(x)) + src_mask * -1000000
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        # output of masked token is always 0
        value = (self.value(self.norm(x)) * (1 - src_mask)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        # y = x + proj((scale+1)*norm(attn(norm(x)))+shift)
        y = x + self.proj_out(y, emb)
        return y


class ChannelWiseSelfAttention(nn.Module):
    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = FiLM(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = self.key(self.norm(x)) + src_mask * -1000000
        query = F.softmax(query.view(B, H, T, -1), dim=-1)
        key = F.softmax(key.view(B, H, T, -1), dim=2)
        # B, T, H, HD
        # output of masked token is always 0
        value = (self.value(self.norm(x)) * (1 - src_mask)).view(B, H, T, -1)
        # B, H, HD, HD
        # B H C T @ B H T C = B H C C
        attention = query.transpose(-1,-2) @ key
        # B H T C @  B H C C= B H C T
        y = (value @ attention).transpose(-1,-2).reshape(B, T, D)
        # y = x + proj((scale+1)*norm(attn(norm(x)))+shift)
        y = x + self.proj_out(y, emb)
        return y



class CrossAttention(nn.Module):
    # 动作为query text为key和value

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = FiLM(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y




