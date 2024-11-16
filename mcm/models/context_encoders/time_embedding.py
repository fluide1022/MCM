import math

import torch
from torch import nn


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    """
        embedding for times step in diffusion model
    """

    def __init__(self, latent_dim, time_embed_dim):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, timestep):
        """
        :param timestep: B
        :return:
        """
        # B C
        timestep = self.time_embed(timestep_embedding(timestep, self.latent_dim)).squeeze(1)
        return timestep
