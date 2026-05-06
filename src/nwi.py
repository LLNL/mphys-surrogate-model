import torch
from torch import nn
from torch.nn import ELU, Identity, Linear, ReLU, Sigmoid, SiLU, Softmax

from src import data_utils as du


# From Huang 2024 https://zenodo.org/records/12866868
class ResBlock(nn.Module):
    def __init__(self, num_features=256):
        super(ResBlock, self).__init__()

        self.stack = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features),
        )
    def forward(self, x):
        return self.stack(x) + x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

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
        nn.LayerNorm(in_features),
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

# From Huang 2025
class NNWF(nn.Module):
    def __init__(self, in_features=64, nodes=16):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(1, nodes),
            ResBlock(num_features=nodes),
            nn.Linear(nodes, 1),
        )
        self.register_buffer(
            "log_bin_mass",
            torch.linspace(-1, 1, in_features)[:, None]
        )
        self.in_features = 1
        self.out_features = 1

    def forward(self):
        return self.stack(self.log_bin_mass) # note: exp moved to LinearEncoder softmax


# With inspiration from Huang 2025
# B = batch size; N = Nbin, L = Nlatent, T = Ntime
# W has size [N, L]
# x has size [B, T, N]
# output has size [B, T, L]
class SimpleNWIEncoder(nn.Module):
    # in this case, W is specified as having no dependence on the bin masses
    def __init__(self, n_bins=64, n_latent=3):
        super(SimpleNWIEncoder, self).__init__()

        self.n_bins = n_bins
        self.n_latent = n_latent

        self.Wlog = nn.Parameter(torch.rand(n_bins, n_latent))

    def wf_mat(self):
        return torch.softmax(self.Wlog, dim=0)

    def forward(self, x):
        return x @ self.wf_mat()

# From Huang2025
class LinearEncoder(nn.Module):
    # in this version, ln(W) = NNWF(ln(m))
    def __init__(self, n_bins=64, n_latent=3):
        super().__init__()
        self.wfs = nn.ModuleList([NNWF(in_features=n_bins) for i in range(n_latent)])
        self.out_features = sum([wf.out_features for wf in self.wfs])

    def wf_mat(self):
        lnW = torch.cat([wf() for wf in self.wfs], dim=1)
        return torch.softmax(lnW, dim=0)

    def forward(self, x):
        return x @ self.wf_mat()


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

# based on Huang 2025
class DeepDecoder(nn.Module):
    def __init__(self, n_bins=64, n_latent=3, hidden_features=256, num_blocks=5, eps=1e-8):
        super().__init__()
        self.stack = DNN(n_latent, n_bins, hidden_features=hidden_features, num_resblocks=num_blocks)
        self.eps = eps
        self.sm = Softmax(dim=-1)
        self.n_bins = n_bins
        self.n_latent = n_latent

    def forward(self, h):
        B = h.shape[0]
        logh = torch.log(h + self.eps)
        logh = logh.reshape(-1, self.n_latent)
        xhat = self.stack(logh)
        xhat = xhat.reshape(B, -1, self.n_bins)
        return self.sm(xhat) # convert back to normalized PSD


class NNWIAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))