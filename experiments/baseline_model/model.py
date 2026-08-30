'''
GPT Model - baseline
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


class Head(nn.Module):
    def __init__(self, T, C, head_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(C, head_size)    # (head_size x C) + head_size learnable params
        self.key = nn.Linear(C, head_size)      # (head_size x C) + head_size learnable params
        self.value = nn.Linear(C, head_size)    # (head_size x C) + head_size learnable params
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))  # (T, T)

    def forward(self, x):

        T = x.shape[1]

        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)              # (B, T, head_size)

        weights = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)                # (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))        # (B, T, T) Dynamic causal-mask slicing
        weights = torch.softmax(weights, dim=-1)                                    # (B, T, T)
        weights = self.dropout(weights)                                             # (B, T, T)

        out = weights @ v    # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, T, C, num_heads, dropout):
        super().__init__()
        assert C % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size = C // num_heads
        self.heads = nn.ModuleList([Head(T, C, self.head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(C, C)            # (C^2) + C  learnable params
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim=-1)
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
        # Weight Sharing Scheme
        self.lm_head.weight = self.embedding.token_embedding.weight
        self._init_weights()

    def _init_weights(module):

        if isinstance(module, nn.Linear):
            nn.init.normal(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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