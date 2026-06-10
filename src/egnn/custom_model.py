"""EGNN predictor used for QM9 regression experiments."""

import torch
from torch import nn


def global_mean_pool_pure(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1
    out = torch.zeros(num_graphs, x.size(-1), device=x.device)
    out.index_add_(0, batch, x)
    count = torch.bincount(batch, minlength=num_graphs).unsqueeze(-1).clamp(min=1)
    return out / count


class EGNNLayer(nn.Module):
    """Single EGNN layer with gated message passing and optional coordinate updates."""

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int = 0,
        m_dim: int = 128,
        update_coords: bool = False,
    ):
        super().__init__()

        self.update_coords = update_coords

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_attr_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.Sigmoid(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + m_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.coor_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1),
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        row, col = edge_index

        x_src = x[row]
        x_dst = x[col]
        relative_position = x_dst - x_src
        distance = relative_position.norm(dim=-1, keepdim=True)

        h_src = h[row]
        h_dst = h[col]

        message_inputs = [h_dst, h_src, distance]
        if edge_attr is not None:
            message_inputs.append(edge_attr)
        message_inputs = torch.cat(message_inputs, dim=-1)

        message = self.edge_mlp(message_inputs)
        gate = self.edge_gate(message)
        message = message * gate

        num_nodes = h.size(0)
        aggregated_message = torch.zeros(num_nodes, message.size(-1), device=h.device)
        aggregated_message.index_add_(0, col, message)

        node_input = torch.cat([h, aggregated_message], dim=-1)
        delta_h = self.node_mlp(node_input)
        h = h + delta_h

        if self.update_coords:
            coordinate_weights = self.coor_mlp(message).squeeze(-1)
            coordinate_updates = relative_position * coordinate_weights.unsqueeze(-1)

            coordinate_delta = torch.zeros_like(x)
            coordinate_delta.index_add_(0, col, coordinate_updates)

            x = x + coordinate_delta

        return h, x


class EGNNQM9Model(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_attr_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 7,
        update_coords: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    m_dim=hidden_dim,
                    update_coords=update_coords,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        h = self.embedding(data.x)
        x = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        mol_repr = global_mean_pool_pure(h, data.batch)
        prediction = self.readout(mol_repr).squeeze(-1)
        return prediction
