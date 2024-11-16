from torch import nn

from models.grad_ops import zero_module


class AudioEncoder(nn.Module):
    def __init__(self, d_sound, d_model):
        super().__init__()

        self.sound_proj = nn.Sequential(
            nn.Linear(d_sound, d_model),
            nn.SiLU(),
            zero_module(nn.Linear(d_model, d_model))
        )

    def forward(self, sound=None):
        return self.sound_proj(sound)
