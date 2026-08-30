'''
Fused QKV 
'''


import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, T, vocab_size, C):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, C)   # (vocab_size x C) learnable params
        self.position_embedding = nn.Embedding(T, C)

    def forward(self, x):
        T = x.shape[1]
        token_emb = self.token_embedding(x)
        pos_embedding = self.position_embedding(torch.arange(T, device=x.device))     
        return token_emb + pos_embedding                           


class MultiHeadAttention(nn.Module):
    def __init__(self, T, C, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        assert C % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size = C // num_heads
        self.qkv = nn.Linear(C, 3 * C)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))  # (T, T)
        self.proj = nn.Linear(C, C)            # (C^2) + C  learnable params
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_size)
        q, k, v = qkv.unbind(dim=2)   # (B, T, num_heads, head_size) x 3
        q = q.transpose(1, 2)   # (B, num_heads, T, head_size)
        k = k.transpose(1, 2)   # (B, num_heads, T, head_size)
        v = v.transpose(1, 2)    # (B, num_heads, T, head_size)
        weights = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)   # (B, num_heads, T, head_size) x (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = torch.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v   # (B, num_heads, T, T) x (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        out = out.transpose(1, 2).contiguous()   # (B, T, num_heads, head_size)
        out = out.view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, C, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4*C),         # (4C^2) + 4C learnable params
            nn.GELU(),
            nn.Linear(4*C, C),         # (4C^2) + C learnable params
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, T, C, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)                                     # 2C learnable params
        self.attn = MultiHeadAttention(T, C, num_heads, dropout)       # (num_heads x (3 x head_size) x (C+1)) + (C^2 + C) learnable params
        self.ln2 = nn.LayerNorm(C)                                     # 2C learnable params
        self.ff = FeedForward(C, dropout)                              # 8C^2 + 5C learnable params 

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, T, C, vocab_size, num_heads, n_layers, dropout):
        super().__init__()
        self.T = T
        self.embedding = Embedding(T, vocab_size, C)
        self.blocks = nn.Sequential(*[Block(T, C, num_heads, dropout) for _ in range(n_layers)])  
        self.ln_f = nn.LayerNorm(C)                 # 2C learnable params
        self.lm_head = nn.Linear(C, vocab_size)     # (vocab_size x C) + vocab_size learnable params
        # Weight Tying
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_tokens=100, temperature=1.0):
        
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(1) <= self.T else idx[:, -self.T:]

            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx