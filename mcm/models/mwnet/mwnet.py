from torch import nn

from mcm.models.mwnet.attention import TimeWiseSelfAttention, CrossAttention, ChannelWiseSelfAttention
from mcm.models.mwnet.ffn import FFN


class MWNetLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1,
                 chan_first=False):
        super().__init__()
        self.sa_block = TimeWiseSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = CrossAttention(
            seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn_1 = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.cwa_block = ChannelWiseSelfAttention(seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ffn_2 = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.chan_first = chan_first

    def forward_chan_post(self, x, xf, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn_1(x, emb)
        x = self.cwa_block(x, emb, src_mask)
        x = self.ffn_2(x, emb)
        return x

    def forward_chan_first(self, x, xf, emb, src_mask):
        x = self.cwa_block(x, emb, src_mask)
        x = self.ffn_2(x, emb)
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn_1(x, emb)
        return x

    def forward(self, x, xf, emb, src_mask):
        if self.chan_first:
            return self.forward_chan_first(x, xf, emb, src_mask)
        return self.forward_chan_post(x, xf, emb, src_mask)


class MWNetNochanLayer(MWNetLayer):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1,
                 chan_first=False):
        super().__init__()
        self.sa_block = TimeWiseSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = CrossAttention(
            seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn_1 = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.cwa_block = TimeWiseSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ffn_2 = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.chan_first = chan_first