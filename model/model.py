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

def format_pytorch_version(version):
    return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
    return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

class CNN_Feat(nn.Module):
    def __init__(self, hidden_dim=348):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        
        for param in list(self.backbone.parameters())[:-5]:
            param.requires_grad = False
    
    def forward(self, X):
        feat = self.backbone(X)
        return feat

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
    
class LSPDecoder(nn.Module):
    def __init__(self, vocab_size, pad_token_id, max_t, hidden_dim = 384, nheads=4, num_decoder_layers=3, dropout=0.4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id

        self.emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos = LearnablePositionalEmbedding(hidden_dim, max_t)

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