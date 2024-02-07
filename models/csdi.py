import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(
            self,
            cond_dim: int,
            hidden_dim: int,
            nheads: int = 4,
        ):
        super().__init__()
        self.temb_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Conv2d(cond_dim, 2 * hidden_dim, 1)
        self.mid_proj = nn.Conv2d(hidden_dim, 2 * hidden_dim, 1)
        self.output_proj = nn.Conv2d(hidden_dim, 2 * hidden_dim, 1)

        time_layer =  nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=64,
            activation="gelu",
        )
        self.time_layer = nn.TransformerEncoder(time_layer, num_layers=1)

        feat_layer =  nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=64,
            activation="gelu",
        )
        self.feat_layer = nn.TransformerEncoder(feat_layer, num_layers=1)

    def forward(self, x: torch.tensor, cond: torch.tensor, temb: torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): [batch_size, hidden_dim, feat_dim, time_dim]
            cond (torch.tensor): [batch_size, cond_dim, feat_dim, time_dim]
            temb (torch.tensor): [batch_size, hidden_dim]

        Returns:
            - torch.tensor: [batch_size, hidden_dim, feat_dim, time_dim]
            - torch.tensor: [batch_size, hidden_dim, feat_dim, time_dim]
        """
        ori_x = x
        temb = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1)
        x = x + temb

        _, _, feat_dim, time_dim = x.shape
        x = rearrange(x, "b h f t -> (b f) t h")
        x = self.time_layer(x)

        x = rearrange(x, "(b f) t h -> (b t) f h", f=feat_dim)
        x = self.feat_layer(x)

        x = rearrange(x, "(b t) f h -> b h f t", t=time_dim)
        x = self.mid_proj(x)

        cond = self.cond_proj(cond)
        x = x + cond

        gate, filter = torch.chunk(x, 2, dim=1)
        x = gate.sigmoid() * filter.tanh()
        x = self.output_proj(x)

        residual, skip = torch.chunk(x, 2, dim=1)
        return (ori_x + residual) / math.sqrt(2.0), skip


class DiffModel(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = 2
        self.hidden_dim = hidden_dim

        self.time_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.input_proj = nn.Sequential(nn.Conv2d(2, self.hidden_dim, 1), nn.ReLU())
        self.compute_block = nn.ModuleList([ResidualBlock(cond_dim, hidden_dim) for i in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, 1, 1),
        )

    def forward(self, x: torch.tensor, cond: torch.tensor, t: torch.tensor):
        """
        Args:
            x: [batch_size, 2, feat, time]
            cond: [batch_size, cond_dim, feat, time]
        Returns:
            - [batch_size, feat, time]
        """
        temb = self.time_embedding(t)
        temb = self.time_proj(temb)

        x = self.input_proj(x)

        skip_lst = []
        for each_layer in self.compute_block:
            x, skip_connection = each_layer(x, cond, temb)
            skip_lst.append(skip_connection)

        x = torch.sum(torch.stack(skip_lst), dim=0) / math.sqrt(len(skip_lst))
        x = self.output_proj(x).squeeze(1)
        return x

    def time_embedding(self, t: torch.tensor):
        """
        Args:
            t: [batch_size, ]
        Returns:
            - [batch_size, self.hidden_dim]
        """
        assert self.hidden_dim % 2 == 0
        dim = self.hidden_dim // 2
        freq = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0).cuda()
        temb = t.unsqueeze(1) * freq
        temb = torch.cat([torch.sin(temb), torch.cos(temb)], dim=1)
        return temb


class CSDI(nn.Module):
    def __init__(
        self,
        num_feat: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.feat_embedding = nn.Parameter(torch.randn(num_feat, self.hidden_dim))
        self.diff_model = DiffModel(hidden_dim*2+1, hidden_dim, n_layers)

    def time_embedding(self, pos):
        pe = torch.zeros(pos.shape[0], pos.shape[1], self.hidden_dim).cuda()
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, self.hidden_dim, 2) / self.hidden_dim
        ).cuda()
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, noisy_data, cond_data, cond_mask, data_t, noise_t):
        """
        Args:
            noisy_data: [batch_size, feat_dim, time_dim]
            cond_data: [batch_size, feat_dim, time_dim]
            cond_mask: [batch_size, feat_dim, time_dim]
            data_t: [batch_size, time_dim]
            noise_t: [batch_size,]

        Returns:
            - [batch_size, feat_dim, time_dim]
        """
        side_info = self.get_side_info(data_t, cond_mask)
        x = torch.stack([noisy_data, cond_data],dim=1)
        score = self.diff_model(x, side_info, noise_t)
        return score
    
    def get_side_info(self, observed_tp, cond_mask):
        num_batch, num_time = observed_tp.shape
        num_feat = self.feat_embedding.shape[0]
        feature_embed = self.feat_embedding.unsqueeze(0).unsqueeze(2).repeat(num_batch,1,num_time,1) # (1,F,1,H)
        feature_embed = rearrange(feature_embed, "b f l h -> b h f l")
        time_embed = self.time_embedding(observed_tp).unsqueeze(1).to(feature_embed).repeat(1,num_feat,1,1) # (B,1,L,H)
        time_embed = rearrange(time_embed, "b f l h -> b h f l")
        side_info = torch.cat([feature_embed, time_embed, cond_mask.unsqueeze(1)], dim=1)
        return side_info
    
if __name__ == "__main__":
    csdi = CSDI(35)
    csdi(torch.randn(16,35,35), torch.randn(16,35,35), torch.randn(16,35,35), torch.randn(16,35), torch.randint(0,20,(16,)))