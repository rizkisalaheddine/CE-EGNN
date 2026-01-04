import torch

from src.egnn_qm9.custom_model import EGNNQM9Model
from src.explainers.cf_explainer_egnn import CFExplainerEGNNRegressionTarget
import argparse
# Example: load one PyG Data object from your dataset
# Replace this with your real dataset import
from torch_geometric.datasets import QM9
from src.egnn_qm9.utils import load_trained_model
from src.egnn_qm9.data import load_qm9_splits, PROPERTY_TO_IDX
import json
import os
import torch

def to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    else:
        return obj


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--property", type=str, default="mu", choices=list(PROPERTY_TO_IDX.keys()))
    parser.add_argument("--molecule_index", type=int, default=8897)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load dataset and pick one molecule ----
    dataset = QM9(root="data/QM9")  # adjust path
    data = dataset[args.molecule_index]              # pick index i you want
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)  # ensure single-graph batch

    # ---- Build model and load trained weights ----
    in_dim = data.x.size(-1)
    edge_attr_dim = 0 if data.edge_attr is None else data.edge_attr.size(-1)

    model_cfg, state = load_trained_model(args.property, device)

    model = EGNNQM9Model(
        in_dim=in_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 7),
        update_coords=model_cfg.get("update_coords", False),
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    # ---- Run counterfactual explainer ----
    explainer = CFExplainerEGNNRegressionTarget(model=model, beta=1e-4, device=device)

    best = explainer.explain(
        data=data,
        alpha=args.alpha,       # target = 1.10 * y_orig
        tau=args.tau,         # tolerance in same units as model output
        lr=1e-2,
        num_epochs=args.epochs,
        grad_clip=2.0,
    )

    if best is None:
        print("No successful CF found (try bigger epochs, lr, or lower beta / larger tau).")
        return

    print("\n=== BEST COUNTERFACTUAL ===")
    for k in ["y_orig", "y_target", "y_cf", "removed_edges", "loss_total", "loss_pred", "loss_graph"]:
        print(f"{k}: {best[k]}")

    # hard_edge_mask_undirected: 1 means kept, 0 means removed
    mask = best["hard_edge_mask_undirected"]
    pairs = best["unique_pairs"]

    removed_pairs = pairs[mask == 0]
    kept_pairs = pairs[mask == 1]

    print("Removed edges (u, v):", removed_pairs[:20])
    print("Kept edges (u, v):", kept_pairs[:20])

    # Save results to JSON
    output_dir = f"outputs/counterfactuals/{args.property}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "molecule_index": args.molecule_index,
        "property": args.property,
        "alpha": args.alpha,
        "tau": args.tau,
        "num_epochs": args.epochs,
        #"counterfactual": to_python(best),
        "y_orig" : best["y_orig"], 
        "y_target": best["y_target"], 
        "y_cf": best["y_cf"], 
        "loss_total": best["loss_total"], 
        "loss_pred":  best["loss_pred"], 
        "loss_graph": best["loss_graph"],
        "removed_edges": removed_pairs.tolist(),
        "kept_edges": kept_pairs.tolist()
    }
    
    with open(f"{output_dir}/cf_{args.property}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
