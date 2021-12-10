import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50
import numpy as np
import pandas as pd
import cv2
import os
import math
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from torchsummary import summary
from tensorboard.plugins import projector
from scipy.optimize import linear_sum_assignment
from utils.utils import *
from utils.assignment import *
from utils.latent_loss import *
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerFast)

def format_pytorch_version(version):
    return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
    return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SinSeeder(nn.Module):
    def __init__(self, n_dim, n_max_sentence):
        super().__init__()

        self.n_dim = n_dim
        pe = torch.zeros(n_max_sentence, n_dim)
        position = torch.arange(0, n_max_sentence,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_dim, 2).float() * (-math.log(10000.0) / n_dim))
        # (t, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, t, dim)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def sample(self, bs, n):
        """
        Returns: (bs, n, dim)
        """
        # (bs, n, dim)
        out = self.pe[:, :n, :].expand(bs, n, self.n_dim)
        return out
    
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (bs, t)
        """
        dev = x.device
        bs, max_t, _ = x.shape
        pos = torch.arange(max_t, dtype=torch.long).unsqueeze(0).expand(bs, max_t).to(device)
        pos = self.emb(pos)
        return x + pos

class SinPositionalEmbedding(nn.Module):
    """
    taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        # (t, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(max_len, d_model)
        self.emb.weight.data = pe
        self.emb.requires_grad_(False)

    def forward(self, x):
        """
        Args:
            x: (bs, t)
        """
        dev = x.device
        bs, max_t, _ = x.shape
        pos = torch.arange(max_t).unsqueeze(0).expand(bs, max_t).to(dev)
        pos = self.emb(pos)
        return x + pos

    def query(self, ids):
        pos = self.emb(ids)
        return pos
    
class SinPositionEmbedding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def query(self, h, w, dev):
        not_mask = torch.ones(1, h, w).to(dev)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=dev)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
class CNN_Feat(nn.Module):
    def __init__(self, hidden_dim=348):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        
        for param in list(self.backbone.parameters())[:-5]:
            param.requires_grad = False
    
    def forward(self, X):
        feat = self.backbone(X)
        return feat

class CNN_Text_SinCos(nn.Module):
    def __init__(self, device, hidden_dim=384, nheads=4, ## According to feature vectors that we get from SBERT, hidden_dim = 384
                 num_encoder_layers=3, num_decoder_layers=3, max_t=20):
        super().__init__()

        # create ResNet-50 backbone
        self.conv_feat = CNN_Feat(hidden_dim)
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create encoder and decoder layers
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads)
        self.decoder = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
        
        # create a default PyTorch transformer: nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder, num_decoder_layers)

        # spatial positional encodings (may be changed to sin positional encodings)
        self.pos = SinPositionEmbedding2D(num_pos_feats=hidden_dim // 2)
        self.seeder = SinSeeder(hidden_dim, max_t)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.device = device
        
    def forward(self, X):
        feat = self.conv_feat(X)
        feat = self.conv(feat)
        bs, c, h, w = feat.shape
        
        ctx = feat + self.pos.query(h, w, X.device)
        ctx = ctx.reshape(bs, c, h * w)
        ctx = ctx.permute([2, 0, 1])
        
        ctx = self.transformer_encoder(ctx)
        
        R = self.seeder.sample(bs, 20).type_as(ctx)
        R = R.permute([1, 0, 2])
        R = self.input_norm(R)
        R = self.transformer_decoder(R, ctx).transpose(0, 1) 
        
        return R, ctx.transpose(0, 1) 
    
    def evaluate(self, X, len_):
        feat = self.conv_feat(X)
        feat = self.conv(feat)
        bs, c, h, w = feat.shape
        
        ctx = feat + self.pos.query(h, w, X.device)
        ctx = ctx.reshape(bs, c, h * w)
        ctx = ctx.permute([2, 0, 1])
        
        ctx = self.transformer_encoder(ctx)
        
        R = self.seeder.sample(bs, len_).type_as(ctx)
        R = R.permute([1, 0, 2])
        R = self.input_norm(R)
        R = self.transformer_decoder(R, ctx).transpose(0, 1) 
        
        return R, ctx.transpose(0, 1) 

    
class CNN_Text(nn.Module):
    def __init__(self, device, hidden_dim=384, nheads=4, ## According to feature vectors that we get from SBERT, hidden_dim = 384
                 num_encoder_layers=3, num_decoder_layers=3):
        super().__init__()

        # create ResNet-50 backbone
        self.conv_feat = CNN_Feat(hidden_dim)
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create encoder and decoder layers
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads)
        self.decoder = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
        
        # create a default PyTorch transformer: nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder, num_decoder_layers)

        # output positional encodings (sentence)
#         self.sentence = nn.Parameter(torch.rand(20, hidden_dim))

        # spatial positional encodings (may be changed to sin positional encodings)
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
        self.device = device
        
    def forward(self, X):
        feat = self.conv_feat(X)
        feat = self.conv(feat)
        H, W = feat.shape[-2:]
        
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        feat = self.transformer_encoder(pos + 0.1 * feat.flatten(2).permute(2, 0, 1))
        R = self.transformer_decoder(torch.rand(20, feat.shape[1], feat.shape[2]).to(self.device), feat).transpose(0, 1) 
        #R = self.transformer_decoder(self.sentence.unsqueeze(1), feat).transpose(0, 1)
        return R, feat
    
    def evaluate(self, X, n_sent):
        feat = self.conv_feat(X)
        feat = self.conv(feat)
        H, W = feat.shape[-2:]
        
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        feat = self.transformer_encoder(pos + 0.1 * feat.flatten(2).permute(2, 0, 1))
        R = self.transformer_decoder(torch.rand(n_sent, feat.shape[1], feat.shape[2]).to(self.device), feat).transpose(0, 1) 
        #R = self.transformer_decoder(self.sentence.unsqueeze(1), feat).transpose(0, 1)
        return R, feat
    
class MLP(nn.Module):
    def __init__(self, hidden_dim=384):
        super().__init__()
        
        self.conv_feat = CNN_Feat(hidden_dim)
        self.linear1 = nn.Linear(131072, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.7)
        
    def forward(self, X):
        feat = self.conv_feat(X)
        x = torch.flatten(feat, 1)
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu()
        x = self.dropout(x)
        output = self.output(x).relu()
        return output
    
class LSPDecoder(PreTrainedModel):
    def __init__(self, vocab_size, pad_token_id, max_t, hidden_dim = 384, nheads=4, num_decoder_layers=3, dropout=0.4):
        super().__init__(PretrainedConfig())
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id

        self.emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos = SinPositionalEmbedding(hidden_dim, max_t)

        self.head = nn.Linear(self.hidden_dim, self.vocab_size)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout),
            num_layers=num_decoder_layers
        )
    
    def loss(self, logits, labels):
        """
        Args:
            logits: (n, t, out)
            labels: (n, t)
            is_empty: (n, )
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()


        shift_labels[shift_labels ==
                         self.pad_token_id] = -100

        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # (n, t)
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss = ce_loss(flat_logits, flat_labels)
        return loss
    
    def forward(self,
                input_ids,
                text_vec,
                context_series,
                labels):
        """
        Args:
            input_ids: teacher forcing signal (bs, t)
            sent_vec: (bs, hid)
            context: (bs, k, hid)
            mems: tuple (current decoded length, transformer state)
        """
        bs, max_t = input_ids.shape
        dev = input_ids.device

        # (bs, t, hid)
        x = self.emb(input_ids)
        # (n, t, n_hid)
        x = self.pos.forward(x)
        if text_vec is not None:
            # (bs, 1, hid)
            text_vec = text_vec.unsqueeze(1)
            x = x + text_vec

        # (t, bs, hid)
        x = x.permute([1, 0, 2])

        # pad mask => (bs, max_t)
        pad_mask = input_ids == self.pad_token_id
        # attention mask
        attn_mask = torch.triu(torch.ones(max_t, max_t).bool(),
                               diagonal=1).to(dev)

        # (t, bs, hid)
        context_series = context_series.permute([1, 0, 2])
        x = self.decoder.forward(tgt=x,
                                 memory=context_series,
                                 tgt_mask=attn_mask,
                                 tgt_key_padding_mask=pad_mask)
        # (bs, t, hid)
        x = x.permute([1, 0, 2])
        # (bs, t, word)
        logits = self.head(x)
        loss = self.loss(logits, labels)

        return loss, logits
    
    def eval_forward(self, input_ids, text_vec, context_series):
        # generate
        # no need to rearrange, we rearrange the loss function instead
        bs, max_t = input_ids.shape
        dev = input_ids.device

        # (bs, t, hid)
        x = self.emb(input_ids)
        # (n, t, n_hid)
        x = self.pos.forward(x)
        if text_vec is not None:
            # (bs, 1, hid)
            text_vec = text_vec.unsqueeze(1)
            x = x + text_vec

        # (t, bs, hid)
        x = x.permute([1, 0, 2])

        # pad mask => (bs, max_t)
        pad_mask = input_ids == self.pad_token_id
        # attention mask
        attn_mask = torch.triu(torch.ones(max_t, max_t).bool(),
                               diagonal=1).to(dev)

        # (t, bs, hid)
        context_series = context_series.permute([1, 0, 2])
        x = self.decoder.forward(tgt=x,
                                 memory=context_series)
        # (bs, t, hid)
        x = x.permute([1, 0, 2])
        # (bs, t, word)
        logits = self.head(x)
        return logits
    
class SentenceEncoder(nn.Module):
    def __init__(self, pad_token_id, n_hid=384, max_t=60, vocab_size=3160, nheads=4, n_layers=3, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_hid)
        self.pos = SinPositionalEmbedding(n_hid, max_t)

        self.input_norm = nn.LayerNorm(n_hid)

        self.encoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_hid, nhead=nheads, dropout=dropout),
            num_layers=n_layers
        )
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, context_series=None, **kwargs):
        """
        Args:
            input_ids: (bs, length)
            context: (bs, k, hid)

        Returns: (bs, n_hid)
        """
        # first word
        
        # (bs, length, n_hid)
        x = self.emb(input_ids) # input_ids
        x = self.pos(x)
        x = self.input_norm(x)
        # (length, bs, n_hid)
        x = x.permute([1, 0, 2])

        pad_mask = input_ids == self.pad_token_id # input_ids
        # (k, bs, hid)
        context_series = context_series.permute([1, 0, 2])

        x = self.encoder.forward(x,
                                 memory=context_series,
                                 tgt_key_padding_mask=pad_mask)
        # (bs, length, n_hid)
        x = x.permute([1, 0, 2])

        # (bs, n_hid)
        head = x[:, 1, :] # 0
        return head
