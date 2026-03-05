import torch
from torch import nn
from torch.nn import ELU, Identity, Linear, ReLU, Sigmoid, SiLU, Softmax

from src import data_utils as du


# From Huang 2024 https://zenodo.org/records/12866868
class ResBlock(nn.Module):
    def __init__(self, num_features=256):
        super(ResBlock, self).__init__()

        self.stack = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features),
        )
    def forward(self, x):
        return self.stack(x) + x

class SimpleBlock(nn.Module):
    def __init__(self, num_features=256):
        super(SimpleBlock, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        return self.stack(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

# From Huang 2024 https://zenodo.org/records/12866868
def DNN(in_features, out_features, hidden_features=256, num_resblocks=5):
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, hidden_features),
        *[ResBlock(hidden_features) for _ in range(num_resblocks)],
        nn.Linear(hidden_features, out_features)
    )

def SNN(in_features, out_features, hidden_features=256, num_blocks=5):
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        *[SimpleBlock(hidden_features) for _ in range(num_blocks)],
        nn.Linear(hidden_features, out_features)
    )


# With inspiration from Huang 2025
# B = batch size; N = Nbin, L = Nlatent, T = Ntime
# W has size [N, L]
# x has size [B, T, N]
# output has size [B, T, L]
class SimpleNWIEncoder(nn.Module):
    def __init__(self, n_bins=64, n_latent=3):
        super(SimpleNWIEncoder, self).__init__()

        self.n_bins = n_bins
        self.n_latent = n_latent

        self.Wlog = nn.Parameter(torch.rand(n_bins, n_latent))

    def wf_mat(self):
        return torch.softmax(self.Wlog, dim=0)

    def forward(self, x):
        return x @ self.wf_mat()

    def to(self, device):
        self.Wlog.to(device)
        return super().to(device)


class SimpleDecoder(nn.Module):
    def __init__(self, n_bins=64, n_latent=3, hidden_features=256, num_blocks=5):
        super(SimpleDecoder, self).__init__()
        self.n_bins = n_bins
        self.n_latent = n_latent

        self.network = SNN(n_latent, n_bins, hidden_features=hidden_features, num_blocks=num_blocks)
        self.sm = Softmax(dim=-1)

    def forward(self, h):
        xhat = self.network(h)
        return self.sm(xhat) # convert back to a normalized PSD


class NNWIAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        return super().to(device)
