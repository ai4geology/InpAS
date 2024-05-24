import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

class VerticalAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(VerticalAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_size, heads)

    def forward(self, Q, K, V, mask):
        return self.multi_head_attention(Q, K, V, mask)

class HorizontalAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(HorizontalAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_size, heads)

    def derive_qkv(self, F):
        # Placeholder for deriving Q, K, V from F
        # Assume F is already transformed appropriately
        return F, F, F

    def reshape_back_to_sequence(self, S_C, shape):
        # Placeholder for reshaping back to original sequence shape
        return S_C.view(shape)

    def forward(self, F, padding_mask):
        Q, K, V = self.derive_qkv(F)
        S_C = self.multi_head_attention(Q, K, V, padding_mask)
        R = self.reshape_back_to_sequence(S_C, F.shape)
        return R

class CombineVerticalAndHorizontalContext(nn.Module):
    def __init__(self, embed_size):
        super(CombineVerticalAndHorizontalContext, self).__init__()
        self.linear_transform = nn.Linear(embed_size * 2, embed_size)

    def forward(self, V_C, H_C):
        C = torch.cat((V_C, H_C), dim=-1)
        CC = self.linear_transform(C)
        return CC

class HC_VR_Attention(nn.Module):
    def __init__(self, embed_size, heads, window_size, step_length, alpha, beta):
        super(HC_VR_Attention, self).__init__()
        self.window_size = window_size
        self.step_length = step_length
        self.alpha = alpha
        self.beta = beta
        self.vertical_attention = VerticalAttention(embed_size, heads)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.combine_contexts = CombineVerticalAndHorizontalContext(embed_size)

    def forward(self, I, M):
        A = []
        # Sliding window logic
        for i in range(0, I.shape[0] - self.window_size + 1, self.step_length):
            for j in range(0, I.shape[1] - self.window_size + 1, self.step_length):
                w = I[:, i:i + self.window_size, j:j + self.window_size]
                mask_w = M[:, i:i + self.window_size, j:j + self.window_size]
                if mask_w.any():
                    w = w * self.alpha
                else:
                    w = w * self.beta
                S = self.positional_encoding(w)
                Q, K, V = S, S, S
                C = self.vertical_attention(Q, K, V, mask=None)
                A.append(C)
        A = torch.cat(A, dim=1)  # Assuming concatenation along the sequence dimension
        return A
