import torch
from torch import nn

from src.egnn.custom_model import EGNNQM9Model, global_mean_pool_pure


def build_undirected_edge_ids(edge_index: torch.Tensor):
    row, col = edge_index
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    pairs = torch.stack([u, v], dim=1)

    unique_pairs, inv = torch.unique(pairs, dim=0, return_inverse=True)
    return pairs, unique_pairs, inv


class EGNNLayerPerturb(nn.Module):
    """EGNN layer variant that applies a learnable edge mask."""

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.update_coords = base_layer.update_coords
        self.edge_mlp = base_layer.edge_mlp
        self.edge_gate = base_layer.edge_gate
        self.node_mlp = base_layer.node_mlp
        self.coor_mlp = base_layer.coor_mlp

    def forward(self, h, x, edge_index, edge_mask, edge_attr=None):
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
        message = message * gate * edge_mask

        num_nodes = h.size(0)
        aggregated_message = torch.zeros(num_nodes, message.size(-1), device=h.device)
        aggregated_message.index_add_(0, col, message)

        node_input = torch.cat([h, aggregated_message], dim=-1)
        h = h + self.node_mlp(node_input)

        if self.update_coords:
            coordinate_weights = self.coor_mlp(message).squeeze(-1)
            coordinate_updates = relative_position * coordinate_weights.unsqueeze(-1)

            coordinate_delta = torch.zeros_like(x)
            coordinate_delta.index_add_(0, col, coordinate_updates)
            x = x + coordinate_delta

        return h, x


class EGNNQM9PerturbRegressionTarget(nn.Module):
    """Counterfactual wrapper that optimizes an undirected edge mask."""

    def __init__(self, base_model: EGNNQM9Model, data):
        super().__init__()

        self.embedding = base_model.embedding
        self.readout = base_model.readout
        self.layers = nn.ModuleList([EGNNLayerPerturb(l) for l in base_model.layers])

        edge_index = data.edge_index
        _, unique_pairs, inv = build_undirected_edge_ids(edge_index)

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_inv", inv)
        self.register_buffer("unique_pairs", unique_pairs)
        self.num_undirected = int(unique_pairs.size(0))

        self.p = nn.Parameter(torch.full((self.num_undirected,), 4.0, device=edge_index.device))
        self.beta = None

    def edge_mask_soft(self, threshold=0.5):
        p_soft_undir = torch.sigmoid(self.p)
        p_hard_undir = (p_soft_undir >= threshold).float()
        p_ste_undir = p_hard_undir + p_soft_undir - p_soft_undir.detach()
        p_dir = p_ste_undir[self.edge_inv]
        return p_dir.unsqueeze(-1)

    @torch.no_grad()
    def edge_mask_hard(self, threshold=0.5):
        hard_undir = (torch.sigmoid(self.p) >= threshold).float()
        hard_dir = hard_undir[self.edge_inv].unsqueeze(-1)
        return hard_undir, hard_dir

    def forward(self, data):
        h = self.embedding(data.x)
        x = data.pos
        edge_attr = data.edge_attr

        edge_mask = self.edge_mask_soft(threshold=0.5)

        for layer in self.layers:
            h, x = layer(h, x, self.edge_index, edge_mask, edge_attr)

        mol_repr = global_mean_pool_pure(h, data.batch)
        return self.readout(mol_repr).squeeze(-1)

    @torch.no_grad()
    def forward_prediction(self, data, threshold=0.5):
        hard_undir, hard_dir = self.edge_mask_hard(threshold=threshold)

        h = self.embedding(data.x)
        x = data.pos
        edge_attr = data.edge_attr

        for layer in self.layers:
            h, x = layer(h, x, self.edge_index, hard_dir, edge_attr)

        mol_repr = global_mean_pool_pure(h, data.batch)
        return self.readout(mol_repr).squeeze(-1), hard_undir

    def loss_regression_target(self, y_cf_soft, y_target, success):
        y_cf_soft = y_cf_soft.squeeze()
        y_target = y_target.squeeze()

        loss_pred = (y_cf_soft - y_target).pow(2)
        loss_graph = torch.sum(1.0 - torch.sigmoid(self.p))
        loss_total = (1.0 - success) * loss_pred + self.beta * loss_graph
        return loss_total, loss_pred, loss_graph
