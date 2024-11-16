"""
Copyright 2021 S-Lab
"""
import copy
from typing import Dict, Optional, Union

import torch
from mmengine import MODELS
from mmengine.model import BaseModel
from torch import nn
import clip

import math

from mcm.models.context_encoders.audio_encoder.audio_encoder import AudioEncoder
from mcm.models.grad_ops import zero_module, set_requires_grad
from mcm.models.mwnet.mwnet import MWNetLayer
from mcm.utils.mask_util import generate_mask


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

@MODELS.register_module(force=True)
class MCM(BaseModel):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu",
                 num_text_layers=4,
                 text_latent_dim=256,
                 text_ff_size=2048,
                 text_num_heads=4,
                 d_audio=4800,
                 use_control=False,
                 chan_first=False,
                 drop_text_rate=0.,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.use_control = use_control
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        self.zero_proj_layers = nn.ModuleList([
            zero_module(nn.Linear(latent_dim, latent_dim)) for _ in range(num_layers)
        ])
        self.drop_text_rate = drop_text_rate
        # Text Transformer
        self.clip, _ = clip.load('ViT-B/32', "cpu")

        set_requires_grad(self.clip, False)
        if text_latent_dim != 512:
            self.text_pre_proj = nn.Linear(512, text_latent_dim)
        else:
            self.text_pre_proj = nn.Identity()
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=dropout,
            activation=activation)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=num_text_layers)
        self.text_ln = nn.LayerNorm(text_latent_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_latent_dim, self.time_embed_dim)
        )

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.main_branch = nn.ModuleList()
        for i in range(num_layers):
            self.main_branch.append(
                MWNetLayer(
                    seq_len=num_frames,
                    latent_dim=latent_dim,
                    text_latent_dim=text_latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                    chan_first=chan_first
                )
            )
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

        if self.use_control:
            self.control_encoder = AudioEncoder(d_audio, self.latent_dim)
            self.control_branch = copy.deepcopy(self.main_branch)
            self.zero_proj_layers = nn.ModuleList([
                zero_module(nn.Linear(latent_dim, latent_dim)) for _ in range(num_layers)
            ])
            self.freeze_main_branch()
        # Output Module

    def freeze_main_branch(self):
        set_requires_grad(self, False)
        set_requires_grad(self.control_encoder, True)
        set_requires_grad(self.control_branch, True)
        set_requires_grad(self.zero_proj_layers, True)

    def encode_text(self, text, device):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, latent_dim]
            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)
        # T, B, D
        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        # B C
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def encode_audio(self, sound=None):
        if not self.use_control:
            return None
        return self.control_encoder(sound)

    def forward_controls(self, h, xf_out, emb, src_mask):
        controls = []
        if not self.use_control:
            return []
        for layer_idx, layer in enumerate(self.control_branch):
            h = layer(h, xf_out, emb, src_mask)
            c = self.zero_proj_layers[layer_idx](h)
            controls.append(c)
        return controls

    def forward_main(self, h, xf_out, emb, src_mask, controls):
        num_uncontrolled_layers = len(self.main_branch) - len(controls)
        for layer_idx, layer in enumerate(self.main_branch):
            if layer_idx < num_uncontrolled_layers:
                h = layer(h, xf_out, emb, src_mask)
            else:
                h = layer(h + controls[layer_idx - num_uncontrolled_layers], xf_out, emb, src_mask)
        return h

    def random_drop_text(self, text):
        return text

    def forward_tensor(self,
                x,
                timesteps,
                length=None,
                text=None,
                xf_proj=None,
                xf_out=None,
                audio=None,
                audio_feature=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        text = self.random_drop_text(text)
        xf_proj, xf_out = self.encode_text(text, x.device) if xf_out is None else (xf_proj, xf_out)
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + xf_proj

        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        # b t 1
        src_mask = generate_mask(T, length).type(torch.float32).to(x.device).unsqueeze(-1)
        controls = []
        if self.use_control:
            assert any([item is not None for item in [audio, audio_feature]])
            audio = self.encode_audio(audio) \
                if audio_feature is None else audio_feature
            controls = self.forward_controls(h + audio, xf_out, emb, src_mask)
        h = self.forward_main(h, xf_out, emb, src_mask, controls)
        output = self.out(h).view(B, T, -1).contiguous()
        return output

    def forward(self,
                inputs: Dict,
                data_samples: Optional[list] = None,
                mode: str = 'loss') -> Union[Dict[str, torch.Tensor], list]:
        if mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'loss':
            return self.forward_loss(inputs, data_samples)
        elif mode == 'tensor':
            return self.forward_tensor(inputs, data_samples)
        else:
            raise NotImplementedError(
                'Forward is not implemented now, please use infer.')

    def forward_loss(self, inputs, data_samples):
        x = inputs['motion']
        B, T, _ = x.shape
        text = inputs['caption']
        num_frames = data_samples.num_frames
        device = x.device

        noise = self.sample_noise(x.shape).to(device)
        timesteps = torch.randint(
            low=0,
            high=self.scheduler.num_train_timesteps, size=(B,),
            device=x.device).long()

        xf_proj, xf_out = self.encode_text(text, x.device)
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + xf_proj

        if self.scheduler.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(x, noise, timesteps)
        elif self.scheduler.prediction_type == 'x_':
            gt = x
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.prediction_type}')




