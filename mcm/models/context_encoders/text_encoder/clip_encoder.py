import torch
from torch import nn
import clip


class CLIP(nn.Module):
    def __init__(self, clip_version: str, max_len_text: int = 77, freeze=True):
        super().__init__()
        self.clip_model = self.load(clip_version)
        if freeze:
            self.freeze()
        self.max_len_text = max_len_text

    def freeze(self):
        # Freeze CLIP weights
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def load(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        return clip_model

    def get_text_embedding(self, text, device, text_feature=None):
        """
        :param text: raw text
        :param text_feature: extracted N L D feature for texts
        :return: N D text embedding
        """
        if text_feature is None:
            return self.encode_text(text, device, is_token=False, return_embedding=True)
        text = clip.tokenize(text, truncate=True, context_length=self.max_len_text).to(device)
        text_feature = text_feature.type(self.clip_model.dtype)
        return text_feature[torch.arange(text_feature.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection

    def encode_text(self, text, device, is_token=False, return_embedding=False):
        if not is_token:
            text = clip.tokenize(text, truncate=True, context_length=self.max_len_text).to(device)
        length = torch.count_nonzero(text, dim=-1)
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_embedding:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection

        return x, length


if __name__ == '__main__':
    raw_text = ['hello world, try clip']
    clip_model = CLIP('ViT-B/32')
    output = clip_model.encode_text(raw_text)
    print(output.shape)
