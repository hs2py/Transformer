import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#Transformer (Attention is All You Need, 2017)


# -----------------------------
# Positional Encoding 
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)   # 홀수 인덱스
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# Scaled Dot-Product Attention
# -----------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn




# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        B, L, D = Q.size()
        q = self.W_q(Q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        out, attn = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.fc(out)


# -----------------------------
# Position-wise Feed Forward
# -----------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# -----------------------------
# Encoder Layer
# -----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + attn_out)         # Residual + Norm
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


# -----------------------------
# Full Transformer Encoder
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)
