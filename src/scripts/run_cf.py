import argparse
import json
import os
import numpy as np
import torch

from src.egnn_qm9.custom_model import EGNNQM9Model
from src.egnn_qm9.data import load_qm9_splits, PROPERTY_TO_IDX
from src.egnn_qm9.utils import load_trained_model
from src.explainers.cf_explainer_egnn import CFExplainerEGNNRegressionTarget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", type=str, default="mu", choices=list(PROPERTY_TO_IDX.keys()))
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=200)
    parser.add_argument("--min_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_graphs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Load deterministic QM9 splits and iterate over test set ----
    (loaders, _, _, _) = load_qm9_splits(
        root="data/QM9",
        batch_size=args.batch_size,
        property_name=args.property,
    )
    _, _, test_loader = loaders

    if args.batch_size != 1:
        raise ValueError("Use --batch_size 1 for per-graph counterfactual generation.")

    test_dataset = test_loader.dataset
    n_test = len(test_dataset)
    limit = n_test if args.max_graphs is None else min(args.max_graphs, n_test)

    # Build model dimensions from first test graph
    sample = test_dataset[0]
    in_dim = sample.x.size(-1)
    edge_attr_dim = 0 if sample.edge_attr is None else sample.edge_attr.size(-1)

    # ---- Build model and load trained weights ----

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

    per_graph = []
    success_mres = []
    num_success = 0

    for idx, data in enumerate(test_loader):
        if idx >= limit:
            break

        best = explainer.explain(
            data=data,
            alpha=args.alpha,
            tau=args.tau,
            lr=1e-2,
            num_epochs=args.epochs,
            eval_every=args.eval_every,
            early_stop_patience=args.early_stop_patience,
            min_epochs=args.min_epochs,
            grad_clip=2.0,
            verbose=args.verbose,
        )

        if best is None:
            per_graph.append(
                {
                    "test_graph_idx": idx,
                    "success": False,
                    "mre": None,
                }
            )
            continue

        y_target = float(best["y_target"])
        y_cf = float(best["y_cf"])
        denom = max(abs(y_target), 1e-12)
        mre = abs(y_cf - y_target) / denom

        mask = best["hard_edge_mask_undirected"]
        pairs = best["unique_pairs"]
        removed_pairs = pairs[mask == 0].tolist()
        kept_pairs = pairs[mask == 1].tolist()

        num_success += 1
        success_mres.append(mre)

        per_graph.append(
            {
                "test_graph_idx": idx,
                "success": True,
                "mre": mre,
                "y_orig": best["y_orig"],
                "y_target": y_target,
                "y_cf": y_cf,
                "loss_total": best["loss_total"],
                "loss_pred": best["loss_pred"],
                "loss_graph": best["loss_graph"],
                "removed_edges_count": int(best["removed_edges"]),
                "removed_edges": removed_pairs,
                "kept_edges": kept_pairs,
            }
        )

        if (idx + 1) % 10 == 0 or idx + 1 == limit:
            print(f"Processed {idx + 1}/{limit} graphs | successes={num_success}")

    mre_mean = float(np.mean(success_mres)) if success_mres else None
    mre_std = float(np.std(success_mres, ddof=0)) if success_mres else None

    # Save results to JSON
    output_dir = f"outputs/counterfactuals/test_{args.property}_a{args.alpha}_tau{args.tau}_e{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "property": args.property,
        "alpha": args.alpha,
        "tau": args.tau,
        "num_epochs": args.epochs,
        "eval_every": args.eval_every,
        "early_stop_patience": args.early_stop_patience,
        "min_epochs": args.min_epochs,
        "num_test_graphs_total": n_test,
        "num_graphs_processed": limit,
        "num_successes": num_success,
        "success_rate": float(num_success / limit) if limit > 0 else 0.0,
        "mre_mean_successes": mre_mean,
        "mre_std_successes": mre_std,
        "per_graph_results": per_graph,
    }

    with open(f"{output_dir}/cf_test_{args.property}.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== TEST SET CF SUMMARY ===")
    print(f"Processed graphs: {limit}/{n_test}")
    print(f"Successes: {num_success}")
    print(f"MRE mean (successes): {mre_mean}")
    print(f"MRE std  (successes): {mre_std}")

if __name__ == "__main__":
    main()
