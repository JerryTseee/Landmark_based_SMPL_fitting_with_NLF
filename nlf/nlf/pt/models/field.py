import numpy as np
import torch
import torch.nn as nn
from simplepyutils import FLAGS

from nlf.paths import PROJDIR


def build_field():
    layer_dims = [FLAGS.field_hidden_size] * FLAGS.field_hidden_layers + [
        (FLAGS.backbone_link_dim + 1) * (FLAGS.depth + 2)
    ]
    gps_net = GPSNet(
        pos_enc_dim=512, hidden_dim=2048, output_dim=FLAGS.field_posenc_dim
    )
    return GPSField(gps_net, layer_dims=layer_dims)


class GPSField(nn.Module):
    def __init__(self, gps_net, layer_dims):
        super().__init__()
        self.posenc_dim = FLAGS.field_posenc_dim
        self.gps_net = gps_net

        # TODO: the first hidden layer's weights should be regularized
        self.pred_mlp = nn.Sequential()
        self.pred_mlp.append(nn.Linear(FLAGS.field_posenc_dim, layer_dims[0]))
        self.pred_mlp.append(nn.GELU())
        for i in range(1, len(layer_dims) - 1):
            self.pred_mlp.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            self.pred_mlp.append(nn.GELU())
        self.pred_mlp.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.r_sqrt_eigva = nn.Buffer(
            torch.rsqrt(
                torch.tensor(np.load(f'{PROJDIR}/canonical_eigval3.npy')[1:], dtype=torch.float32)
            ),
            persistent=False,
        )

    def forward(self, inp):
        lbo = self.gps_net(inp.reshape(-1, 3))[..., : self.posenc_dim]
        lbo = torch.reshape(lbo, inp.shape[:-1] + (self.posenc_dim,))
        lbo = lbo * self.r_sqrt_eigva[: self.posenc_dim] * 0.1
        return self.pred_mlp(lbo)


class GPSNet(nn.Module):
    def __init__(self, pos_enc_dim=512, hidden_dim=2048, output_dim=1024):
        super().__init__()
        self.factor = 1 / np.sqrt(np.float32(pos_enc_dim))
        nodes = np.load(f'{PROJDIR}/canonical_nodes3.npy')
        self.mini = nn.Buffer(
            torch.tensor(np.min(nodes, axis=0), dtype=torch.float32), persistent=False
        )
        self.maxi = nn.Buffer(
            torch.tensor(np.max(nodes, axis=0), dtype=torch.float32), persistent=False
        )
        self.center = nn.Buffer((self.mini + self.maxi) / 2, persistent=False)

        self.learnable_fourier = LearnableFourierFeatures(3, pos_enc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inp):
        x = (inp - self.center) / (self.maxi - self.mini)
        x = self.learnable_fourier(x) * self.factor
        return self.mlp(x)


class LearnableFourierFeatures(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError('out_features must be even (sin and cos in pairs)')
        self.linear = nn.Linear(in_features, out_features // 2, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=12)

    def forward(self, inp):
        x = self.linear(inp)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
