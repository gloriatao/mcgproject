''' Define the Transformer model '''
import torch, math
import torch.nn as nn
import numpy as np
from models.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
from models.position_encoding import build_position_encoding

__author__ = "modified from Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False

class NLWapperEncoder(nn.Module):
    def __init__(self, n_layers, hidden, max_position_embeddings=1024):
        super().__init__()
        self.hidden = hidden
        self.encoder = Encoder(n_layers=n_layers, n_head=8, d_model=hidden, d_inner=1024, dropout=0.1)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden)
        create_sinusoidal_embeddings(n_pos=max_position_embeddings, dim=hidden, out=self.position_embeddings.weight)
        self.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        bs, seq_length, dim = x.shape
        # tag = (torch.ones((bs, 1, dim))*0.2).to(x.device)
        # x = torch.cat((tag, x), dim=1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)  # (max_seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings = position_embeddings.unsqueeze(0).expand_as(x)  # (bs, max_seq_length)

        embeddings = x + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)
        # embeddings = embeddings.permute(0,2,1) # (bs, dim,max_seq_length)

        out, attn = self.encoder(embeddings, None)

        return out, attn

class NLWapperDecoder(nn.Module):
    def __init__(self, d_model, n_layers, n_queries):
        super().__init__()
        self.hidden = d_model
        self.n_queries = n_queries
        self.decoder = Decoder(n_layers=n_layers, d_model=d_model)
        self.decoder_position_embedding = nn.Embedding(n_queries, d_model)
        self.encoder_position_embedding = build_position_encoding(d_model // 3, type='sine')

    def forward(self, enc_output, decoder_self_mask, padding_mask, dec_output=None):
        if enc_output.shape[-1] > self.max_upsampling_size[-1]:
            enc_output = F.adaptive_max_pool3d(enc_output, self.max_upsampling_size)

        padding_mask = F.interpolate(padding_mask[None].float(), size=enc_output.shape[-3:]).to(torch.bool)[0]
        pos_embedding = self.encoder_position_embedding(enc_output, padding_mask).to(enc_output.dtype)
        enc_output = enc_output + pos_embedding

        bs, c, w, h, d = enc_output.shape
        if dec_output == None:
            dec_output = torch.zeros((bs, self.n_queries, self.hidden)).to(enc_output.device)

        decoder_out, d_self_attn, d_cross_attn = self.decoder(dec_output, enc_output, decoder_self_mask, padding_mask)

        return decoder_out, d_self_attn, d_cross_attn

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    # Decoder = DecoderLayer(d_model=384, d_inner=2048, n_head=8, d_k=48, d_v=48)
    # Encoder = EncoderLayer(d_model=384, d_inner=2048, n_head=8, d_k=48, d_v=48, dropout=0.1)

    def __init__(self, n_layers, n_head, d_model, d_inner, dropout):
        super().__init__()
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        d_k = d_model//n_head
        d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask):

        enc_slf_attn_list = []

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(src_seq, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn]

        return enc_output, enc_slf_attn_list

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self, n_layers=1, n_head=8, d_model=512, d_inner=1024, dropout=0.1):
        super().__init__()

        d_k = d_model//n_head
        d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_output,  enc_output, decoder_self_mask, padding_mask):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        bs, c, w, h, d = enc_output.shape
        class_num = dec_output.shape[1]
        enc_output = enc_output.view(bs, c, -1).permute(0,2,1)

        # # -- Forward
        # dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))
        # dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_enc_attn_mask = torch.zeros(bs, class_num, w*h*d).to(dec_output.device)
            for b in range(bs):
                p = padding_mask[b].unsqueeze(0)
                encoder_mask = F.interpolate(p[None].float(), size=(w,h,d)).to(torch.bool)[0]
                encoder_mask = encoder_mask.flatten(1)

                dec_enc_attn_mask[b, :, :] = encoder_mask.unsqueeze(1).repeat(1, class_num, 1)

            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=decoder_self_mask, dec_enc_attn_mask=dec_enc_attn_mask)
            dec_slf_attn_list += [dec_slf_attn]
            dec_enc_attn_list += [dec_enc_attn]

        return dec_output, dec_slf_attn_list, dec_enc_attn_list,


class Seghead(nn.Module):
    def __init__(self, layer_stack1=[192,96,48,24,12], n_hidden=384, n_heads=8):
        super().__init__()
        self.layers1 = nn.ModuleList([torch.nn.Conv3d(n_hidden + 11, r, 3, padding=1) for i, r in enumerate(layer_stack1)])
        layer_stack_previous = layer_stack1[0:-1]
        layer_stack_current = layer_stack1[1:]
        self.layers2 = nn.ModuleList([torch.nn.Conv3d(layer_stack_previous[i] + r, r, 3, padding=1) for i, r in enumerate(layer_stack_current)])

        self.gns1 = nn.ModuleList([torch.nn.GroupNorm(4, r) for r in (layer_stack1)])
        self.gns2 = nn.ModuleList([torch.nn.GroupNorm(4, r) for r in (layer_stack_current)])
        print()

    def forward(self, dec_output, decoder_cross_attn, enc_output,level,seg_out=None):
        ratio = int(enc_output.shape[-1]*enc_output.shape[-2]*enc_output.shape[-3]/decoder_cross_attn.shape[-1])
        r_ratio = round(math.pow(ratio,1/3))
        bs, head, c, _ = decoder_cross_attn.shape
        _,_,w,h,d = enc_output.shape
        decoder_cross_attn = decoder_cross_attn.view(bs,head,c,w//r_ratio,h//r_ratio,d//r_ratio).mean(dim=1)
        decoder_cross_attn = F.interpolate(decoder_cross_attn, scale_factor=r_ratio)
        out = torch.cat([enc_output, decoder_cross_attn], 1)

        if level==0:
            out = self.layers1[level](out)
            out = self.gns1[level](out)
            out = F.relu(out)
        else:
            seg_out = F.interpolate(seg_out, scale_factor=2, mode='trilinear', align_corners=False)
            out = self.layers1[level](out)
            out = self.gns1[level](out)
            out = F.relu(out)
            out = torch.cat([seg_out, out],1)
            out = self.layers2[level-1](out)
            out = self.gns2[level-1](out)
            out = F.relu(out)

        return out

class Seghead_old(nn.Module):
    def __init__(self, layer_stack1=[32,16,16,8,8], n_hidden=384,n_heads=8):
        super().__init__()
        # trans = [i//2 for i in inters]
        # self.layers = nn.ModuleList([torch.nn.Conv3d(d_hidden+n_heads, inters[0]//2, 3, padding=1)] + [torch.nn.Conv3d(d_hidden+n_heads+trans[i], r//2, 3, padding=1) for i, r in enumerate(inters)])
        self.layers1 = nn.ModuleList([torch.nn.Conv3d(n_hidden + n_heads, r, 3, padding=1) for i, r in enumerate(layer_stack1)])
        layer_stack_previous = layer_stack1[0:-1]
        layer_stack_current = layer_stack1[1:]
        self.layers2 = nn.ModuleList([torch.nn.Conv3d(layer_stack_previous[i] + r, r, 3, padding=1) for i, r in enumerate(layer_stack_current)])

        self.gns1 = nn.ModuleList([torch.nn.GroupNorm(4, r) for r in (layer_stack1)])
        self.gns2 = nn.ModuleList([torch.nn.GroupNorm(4, r) for r in (layer_stack_current)])
        print()

    def forward(self, dec_output, decoder_cross_attn, enc_output,level,seg_out=None):
        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1, 1).flatten(0, 1)

        tmp = expand(enc_output, dec_output.shape[1])
        bs, head, c, _ = decoder_cross_attn.shape
        _,_,w,h,d = tmp.shape
        decoder_cross_attn = decoder_cross_attn.view(bs,head,c,w,h,d).flatten(0, 1).permute(1,0,2,3,4)
        out = torch.cat([tmp, decoder_cross_attn], 1)

        if level==0:
            out = self.layers1[level](out)
            out = self.gns1[level](out)
            out = F.relu(out)
        else:
            seg_out = F.interpolate(seg_out, scale_factor=2, mode='trilinear', align_corners=False)
            out = self.layers1[level](out)
            out = self.gns1[level](out)
            out = F.relu(out)
            out = torch.cat([seg_out, out],1)
            out = self.layers2[level-1](out)
            out = self.gns2[level-1](out)
            out = F.relu(out)

        return out



