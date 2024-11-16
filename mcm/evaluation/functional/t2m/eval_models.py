from typing import Tuple, Union, List

import torch
import torch.nn as nn
import numpy as np
import math
from mmengine import MODELS
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def reparameterize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)


# batch_size, dimension and position
# output: (batch_size, dim)
def positional_encoding(batch_size, dim, pos):
    assert batch_size == pos.shape[0]
    positions_enc = np.array([
        [pos[j] / np.power(10000, (i - i % 2) / dim) for i in range(dim)]
        for j in range(batch_size)
    ], dtype=np.float32)
    positions_enc[:, 0::2] = np.sin(positions_enc[:, 0::2])
    positions_enc[:, 1::2] = np.cos(positions_enc[:, 1::2])
    return torch.from_numpy(positions_enc).float()


def get_padding_mask(batch_size, seq_len, cap_lens):
    cap_lens = cap_lens.data.tolist()
    mask_2d = torch.ones((batch_size, seq_len, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(cap_lens):
        mask_2d[i, :, :cap_len] = 0
    return mask_2d.bool(), 1 - mask_2d[:, :, 0].clone()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class TextVAEDecoder(nn.Module):
    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextVAEDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        #
        # self.output = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(hidden_size, output_size-4)
        # )

        # self.contact_net = nn.Sequential(
        #     nn.Linear(output_size-4, 64),
        #     nn.LayerNorm(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64, 4)
        # )

        self.output.apply(init_weight)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.contact_net.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs, last_pred, hidden, p):
        h_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).to(inputs.device).detach()
        h_in = h_in + pos_enc
        for i in range(self.n_layers):
            # print(h_in.shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        # pose_pred = self.output(h_in) + last_pred.detach()
        # contact = self.contact_net(pose_pred)
        # return torch.cat([pose_pred, contact], dim=-1), hidden
        return pose_pred, hidden


class TextDecoder(nn.Module):
    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.positional_encoder = PositionalEncoding(hidden_size)

        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)

        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        self.mu_net.apply(init_weight)
        self.logvar_net.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)

        return list(hidden)

    def forward(self, inputs, hidden, p):
        # print(inputs.shape)
        x_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).to(inputs.device).detach()
        x_in = x_in + pos_enc

        for i in range(self.n_layers):
            hidden[i] = self.gru[i](x_in, hidden[i])
            h_in = hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = reparameterize(mu, logvar)
        return z, mu, logvar, hidden


class AttLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)

        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim

        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, query, key_mat):
        '''
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        '''
        # print(query.shape)
        query_vec = self.W_q(query).unsqueeze(-1)  # (batch, value_dim, 1)
        val_set = self.W_v(key_mat)  # (batch, seq_len, value_dim)
        key_set = self.W_k(key_mat)  # (batch, seq_len, value_dim)

        weights = torch.matmul(key_set, query_vec) / np.sqrt(self.dim)

        co_weights = self.softmax(weights)  # (batch, seq_len, 1)
        values = val_set * co_weights  # (batch, seq_len, value_dim)
        pred = values.sum(dim=1)  # (batch, value_dim)
        return pred, co_weights

    def short_cut(self, querys, keys):
        return self.W_q(querys), self.W_k(keys)


class TextEncoderBiGRU(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, device):
        super(TextEncoderBiGRU, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        gru_seq = pad_packed_sequence(gru_seq, batch_first=True)[0]
        forward_seq = gru_seq[..., :self.hidden_size]
        backward_seq = gru_seq[..., self.hidden_size:].clone()

        # Concate the forward and backward word embeddings
        for i, length in enumerate(cap_lens):
            backward_seq[i:i + 1, :length] = torch.flip(backward_seq[i:i + 1, :length].clone(), dims=[1])
        gru_seq = torch.cat([forward_seq, backward_seq], dim=-1)

        return gru_seq, gru_last


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True, enforce_sorted=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionLenEstimatorBiGRU(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super(MotionLenEstimatorBiGRU, self).__init__()

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output(gru_last)


def build_eval_models(dim_pose: int = 263,
                      dim_motion_hidden: int = 1024,
                      dim_movement_enc_hidden: int = 512,
                      dim_movement_latent: int = 512,
                      dim_word: int = 300,
                      dim_pos_ohot: int = 15,
                      dim_text_hidden: int = 512,
                      dim_coemb_hidden: int = 512,
                      eval_model: str = 'pretrained_llama/humanml3d/text_mot_match/model/finest.tar',
                      device: Union[str, torch.device] = 'cpu'
                      ):
    movement_enc = MovementConvEncoder(dim_pose - 4, dim_movement_enc_hidden, dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=dim_text_hidden,
                                  output_size=dim_coemb_hidden,
                                  device=device)

    motion_enc = MotionEncoderBiGRUCo(input_size=dim_movement_latent,
                                      hidden_size=dim_motion_hidden,
                                      output_size=dim_coemb_hidden,
                                      device=device)

    checkpoint = torch.load(eval_model, map_location=device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}


@MODELS.register_module(force=True)
class T2MExtractor(object):

    def __init__(self,
                 dim_pose: int = 263,
                 unit_length=4,
                 dim_motion_hidden=1024,
                 dim_movement_enc_hidden: int = 512,
                 dim_movement_latent: int = 512,
                 dim_word: int = 300,
                 dim_text_hidden: int = 512,
                 dim_coemb_hidden: int = 512,
                 eval_model: str = 'checkpoints/humanml3d/text_mot_match/model/finest.tar',
                 device: Union[str, torch.device] = 'cpu',
                 max_text_len: int = 20,
                 **kwargs
                 ):
        """
        :param dim_pose:
        :param unit_length:
        :param dim_movement_enc_hidden:
        :param dim_movement_latent:
        :param dim_word:
        :param dim_text_hidden:
        :param dim_coemb_hidden: dimension of
        :param eval_model: pretrained model path
        :param device: device of eval_datasets models
        """
        dim_pos_ohot = len(POS_enumerator)
        self.max_text_len = max_text_len
        self.text_encoder, self.motion_encoder, self.movement_encoder = build_eval_models(
            dim_pose=dim_pose,
            dim_motion_hidden=dim_motion_hidden,
            dim_movement_enc_hidden=dim_movement_enc_hidden,
            dim_movement_latent=dim_movement_latent,
            dim_word=dim_word,
            dim_pos_ohot=dim_pos_ohot,
            dim_text_hidden=dim_text_hidden,
            dim_coemb_hidden=dim_coemb_hidden,
            eval_model=eval_model
        )
        self.device = device
        self.unit_length = unit_length
        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs: torch.Tensor,
                          pos_ohot: torch.Tensor,
                          cap_lens: torch.Tensor,
                          motions: torch.Tensor,
                          m_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param word_embs: word_embs extracted by word vectorizer
        :param pos_ohot: one-hot vector of parts of speech
        :param cap_lens: caption length
        :param motions: motion vectors in humanml3d format
        :param m_lens: motion length
        :return: extracted text embedding and motion embedding
        """

        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()

            m_lens = m_lens // self.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        """
        :param motions: humanml3d format vectors
        :param m_lens: timewise length of each motion vector
        :return: extracted motion embeddings. Same as extracted in get_co_embeddings.
        """
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

    def process_text(self, text: str) -> Tuple[List[str], List[str]]:
        """ borrowed from MotionGPT, thanks for the code
        get word list and pos list of a raw sentence
        :param text: raw text
        :return: word_list, pos_list
        """
        text = text.replace('-', '')
        doc = self.nlp(text)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN'
                or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """ Borrowed from motion_gpt. Thanks for the code.
        :param texts: batch of texts
        :return: text embeddings
        """
        word_embs = []
        pos_ohot = []
        text_lengths = []
        for i, sentence in enumerate(texts):
            word_list, pos_list = self.process_text(sentence.strip())
            t_tokens = [
                '%s/%s' % (word_list[i], pos_list[i])
                for i in range(len(word_list))
            ]

            if len(t_tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'
                                   ] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = t_tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(torch.tensor(pos_oh).float()[None])
                word_embeddings.append(torch.tensor(word_emb).float()[None])
            text_lengths.append(sent_len)
            pos_ohot.append(torch.cat(pos_one_hots, dim=0)[None])
            word_embs.append(torch.cat(word_embeddings, dim=0)[None])

        word_embs = torch.cat(word_embs, dim=0).to(self.Matching_score)
        pos_ohot = torch.cat(pos_ohot, dim=0).to(self.Matching_score)
        text_lengths = torch.tensor(text_lengths).to(self.Matching_score)

        align_idx = np.argsort(text_lengths.data.tolist())[::-1].copy()

        # get text embeddings
        text_embeddings = self.text_encoder(word_embs[align_idx],
                                            pos_ohot[align_idx],
                                            text_lengths[align_idx])

        original_text_embeddings = text_embeddings.clone()

        for idx, sort in enumerate(align_idx):
            original_text_embeddings[sort] = text_embeddings[idx]

        return original_text_embeddings
