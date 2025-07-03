from math import log, pi
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


class ScanIDEmbedding(nn.Module):
    """
    Scan ID Embedding:
    - Given a list of scan IDs, return the embeddings of the scan IDs.
    """
    def __init__(self, dim, max_unique_scans=64):
        super().__init__()
        self.dim = dim
        self.max_num = max_unique_scans
        self.embd = nn.Embedding(max_unique_scans, dim)

    def forward(self, file_list):
        files_unique = np.unique(file_list)
        files_unique.sort()  # the reference array
        indices = np.searchsorted(files_unique, file_list)
        device = self.embd.weight.device
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        return self.embd(indices)


class PositionEmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, normalizer_in_mm=150, dropout=0.):
        super(PositionEmbeddingMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()  # Nonlinearity
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)
        self.normalizer_in_mm = normalizer_in_mm

    def forward(self, x):
        x = (x - x.mean(dim=0, keepdim=True)) / self.normalizer_in_mm
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    dim = 768
    max_unique_scans = 64
    # Test ScanIDEmbedding
    embd = ScanIDEmbedding(dim=dim, max_unique_scans=max_unique_scans)
    embd = embd.cuda()
    file_list = np.random.randint(0, 64, 100)
    e_scan = embd(torch.tensor(file_list))

    # Test PositionEmbeddingMLP
    embd = PositionEmbeddingMLP(32, 64, 768)
    embd.cuda()
    x = torch.randn(16, 32).cuda()
    e_pos = embd(x)

