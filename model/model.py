import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import regularization

class VMASK(nn.Module):
    def __init__(self, embed_dim, mask_hidden_dim, activation='tanh'):
        super(VMASK, self).__init__()
        activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu}
        self.activation = activations[activation]
        self.linear_layer = nn.Linear(embed_dim, mask_hidden_dim)
        self.hidden2p = nn.Linear(mask_hidden_dim, 2)
        self.mask = None

    def forward_sent_batch(self, embeds):
        p = self.hidden2p(self.activation(self.linear_layer(embeds)))
        return p

    def forward(self, x, p, flag):
        if flag == 'train':
            r = F.gumbel_softmax(p, hard=True, dim=2)[:, :, 1:2]
            self.mask = r
            return r * x
        else:
            probs = F.softmax(p, dim=2)[:, :, 1:2]
            self.mask = probs
            return x * probs

    def get_statistics_batch(self, embeds):
        return self.forward_sent_batch(embeds)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key   = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out(context), attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x, attn_weights

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,
                 num_classes, dropout, mask_hidden_dim):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.vmask = VMASK(d_model, mask_hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.reverse_vocab = None

    def set_reverse_vocab(self, vocab):
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    def generate_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def forward(self, src, flag):
        if self.reverse_vocab is not None:
            self.current_tokens = [self.reverse_vocab.get(idx.item(), '<UNK>') for idx in src[0]]
        mask = self.generate_mask(src)
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        self.embed = x

        # Apply VMASK
        p = self.vmask.get_statistics_batch(x)
        self.p = p
        x = self.vmask(x, p, flag)
        self.x_vmask = x

        x = self.pos_encoder(x)
        x = self.dropout(x)

        all_attn_weights = []
        for layer in self.layers: # only one layer in our task setting
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)
        self.all_attn_weights = torch.stack(all_attn_weights)
        self.x_ffn = x
        self.attn_weights_one_layer = self.all_attn_weights.squeeze(0)
        self.attn_correlation = self.attn_weights_one_layer / self.attn_weights_one_layer.sum(dim=-1, keepdim=True)
        x = x.mean(dim=1)
        x = self.fc(x)
        self.output = x

        # Calculate regularization losses using functions from regularization.py
        self.polyse_loss = regularization.polysemanticity_loss(self.x_vmask)
        self.inter_loss = regularization.correlation_interference_loss(self.attn_weights_one_layer, self.attn_correlation)
        return x
