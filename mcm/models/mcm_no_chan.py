"""
Copyright 2021 S-Lab
"""
import copy
import torch
from mmengine import MODELS
from torch import nn
import clip
from torch.nn import Module

from models import MCM
from models.context_encoders.audio_encoder.audio_encoder import AudioEncoder
from models.grad_ops import zero_module, set_requires_grad
from models.mwnet.mwnet import MWNetNochanLayer
@MODELS.register_module(force=True)
class MCMNochan(MCM):
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
        Module.__init__(self)

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
                MWNetNochanLayer(
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
