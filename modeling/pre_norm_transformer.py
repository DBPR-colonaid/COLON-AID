import torch
import torch.nn as nn

class PreNorm(nn.Module):
    """ Helper class for Pre-Layer Normalization """
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class FeedForward(nn.Module):
    """ Feed-forward network as used in Transformer encoder layers """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """ Multi-head self-attention module """
    def __init__(self, dim, heads=8, dropout=0.1):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_q_attn=True):
        b, n, _, h = *x.shape, self.heads
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)  # (b, h, n, d)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, h, n, n)
        attn = dots.softmax(dim=-1)
        q_attn = None
        if return_q_attn:
            # get the attention matrix w.r.t. the cls token
            q_attn = attn[:, :, 0, :]  # (b, h, n)
            q_attn = q_attn.cpu()

        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        if return_q_attn:
            return self.proj(out), q_attn
        return self.proj(out)

class TransformerEncoderLayer(nn.Module):
    """ Single Transformer Encoder Layer with Pre-Layer-Norm """
    def __init__(self, dim, heads, ff_hidden_mult=4, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = PreNorm(dim, Attention(dim, heads, dropout))
        self.feed_forward = PreNorm(dim, FeedForward(dim, dim * ff_hidden_mult, dropout))

    def forward(self, x):
        attn_res, q_attn = self.attention(x)
        x = x + attn_res
        x = x + self.feed_forward(x)
        return x, q_attn

class TransformerEncoder(nn.Module):
    """ Full Transformer Encoder with multiple layers """
    def __init__(self, dim, depth, heads, ff_hidden_mult=4, dropout=0.1, return_q_attn=False):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, heads, ff_hidden_mult, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.return_q_attn = return_q_attn

    def forward(self, x):
        q_attn = None
        for layer in self.layers:
            x, q_attn = layer(x)
        if self.return_q_attn:
            return self.norm(x), q_attn
        return self.norm(x)


if __name__ == '__main__':
    # Example initialization and usage
    dim = 512
    depth = 6
    heads = 8
    model = TransformerEncoder(dim, depth, heads, return_q_attn=1)
    x = torch.randn(10, 20, dim)  # Example input: batch_size=10, seq_length=20, features=dim
    out, attn = model(x)
    print(out.shape)
    print(attn.shape)
