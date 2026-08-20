import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, T, C):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, C)   # (vocab_size, C) learnable params
        self.position_embedding = nn.Embedding(T, C)         # (T, C) learnable params

    def forward(self, x):
        B, T = x.shape
        token_emb = self.token_embedding(x)                         # (B, T, C)
        position_emb = self.position_embedding(torch.arange(T))     # (T, C)
        return token_emb + position_emb                             # (B, T, C)


class Head(nn.Module):
    def __init__(self, T, C, head_size):
        super().__init__()
        self.T = T
        self.query = nn.Linear(C, head_size)    # (head_size, C)  learnable params
        self.key = nn.Linear(C, head_size)      # (head_size, C)  learnable params
        self.value = nn.Linear(C, head_size)    # (head_size, C)  learnable params
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))  # (T, T)

    def forward(self, x):

        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        weights = q @ k.transpose(-2, -1)                                                # (B, T, T)
        weights = weights.masked_fill(self.tril[:self.T, :self.T] == 0, float('-inf'))   # (B, T, T)
        weights = torch.softmax(weights)                                                 # (B, T, T)

        out = weights @ v    # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, T, C, num_heads):
        super().__init__()
        self.head_size = C // num_heads
        self.heads = nn.ModuleList([Head(T, C, self.head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(C, C)     # (C, C) learnable params

    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4*C),         # (4*C, C) learnable params
            nn.ReLU(),
            nn.Linear(4*C, C)          # (C, 4*C) learnable params
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, T, C, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)                       # (2 * C) learnable params
        self.attn = MultiHeadAttention(T, C, num_heads)
        self.ln2 = nn.LayerNorm(C)                       # (2 * C) learnable params
        self.ff = FeedForward(C)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, T, C, vocab_size, num_heads, n_layers, dropout):
        super().__init__()
        self.T = T
        self.embedding = EmbeddingLayer(vocab_size, T, C)
        self.blocks = nn.Sequential(*[Block(T, C, num_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size)     # (vocab_size, C) learnable params

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

            probs = torch.softmax(logits)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx