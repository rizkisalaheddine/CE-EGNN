import argparse
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch


ATOM_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}

ATOM_COLOR = {
    1: "#dddddd",
    6: "#444444",
    7: "#4f83cc",
    8: "#d33f49",
    9: "#8ec07c",
}


def _as_edge_list(pairs):
    return [tuple(map(int, p)) for p in pairs]


def _build_graph(num_nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    return graph


def _draw_graph(ax, graph, pos, labels, node_colors, title, highlight_edges=None):
    ax.set_title(title)
    nx.draw_networkx_nodes(graph, pos, node_size=400, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, ax=ax)
    if highlight_edges:
        kept = highlight_edges.get("kept", [])
        removed = highlight_edges.get("removed", [])
        if kept:
            nx.draw_networkx_edges(graph, pos, edgelist=kept, width=1.5, ax=ax)
        if removed:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=removed,
                width=2.0,
                edge_color="#d33f49",
                style="dashed",
                ax=ax,
            )
    else:
        nx.draw_networkx_edges(graph, pos, width=1.5, ax=ax)
    ax.axis("off")


def _load_qm9_geometry(processed_path, index):
    if not processed_path:
        raise ValueError("processed_path is required")
    data, slices = torch.load(processed_path)
    if "pos" not in slices or "z" not in slices:
        raise KeyError("processed file missing pos/z slices")
    pos = data.pos[slices["pos"][index]:slices["pos"][index + 1]]
    z = data.z[slices["z"][index]:slices["z"][index + 1]]
    return pos, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", required=True, help="Path to cf_*.json")
    parser.add_argument(
        "--processed",
        default="data/QM9/processed/data_v3.pt",
        help="Path to processed QM9 data file",
    )
    parser.add_argument("--save", default="", help="Optional output image path")
    args = parser.parse_args()

    with open(args.cf, "r") as f:
        cf = json.load(f)

    pos, z = _load_qm9_geometry(args.processed, cf["molecule_index"])
    pos = pos[:, :2].cpu().numpy()
    num_nodes = pos.shape[0]
    z = z.cpu().tolist()

    labels = {i: ATOM_SYMBOL.get(z_i, str(z_i)) for i, z_i in enumerate(z)}
    node_colors = [ATOM_COLOR.get(z_i, "#aaaaaa") for z_i in z]

    kept_edges = _as_edge_list(cf.get("kept_edges", []))
    removed_edges = _as_edge_list(cf.get("removed_edges", []))
    original_edges = list({tuple(sorted(e)) for e in kept_edges + removed_edges})

    graph_orig = _build_graph(num_nodes, original_edges)
    graph_cf = _build_graph(num_nodes, kept_edges)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _draw_graph(
        axes[0],
        graph_orig,
        pos,
        labels,
        node_colors,
        title="Original (removed edges in red)",
        highlight_edges={"kept": kept_edges, "removed": removed_edges},
    )
    _draw_graph(
        axes[1],
        graph_cf,
        pos,
        labels,
        node_colors,
        title="Counterfactual",
    )
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
