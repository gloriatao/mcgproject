''' Define the Layers '''
import torch.nn as nn
import torch
from models.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "modified from Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


# def test():
#     Decoder = DecoderLayer(d_model=384, d_inner=2048, n_head=8, d_k=48, d_v=48)
#     Encoder = EncoderLayer(d_model=384, d_inner=2048, n_head=8, d_k=48, d_v=48, dropout=0.1)
#
#     # input = torch.randn([1,216, 384])
#     # input = torch.tensor(input, dtype=torch.float32)
#     # y = Encoder(input)
#
#     enc_output = torch.randn([1, 216, 384])
#     enc_output = torch.tensor(enc_output, dtype=torch.float32)
#
#     dec_input = torch.randn([1, 25, 384])
#     dec_input = torch.tensor(dec_input, dtype=torch.float32)
#     y = Decoder(dec_input, enc_output)
#     return

# test()
