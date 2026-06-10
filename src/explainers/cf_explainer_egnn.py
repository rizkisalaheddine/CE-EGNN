import torch
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from .egnn_perturb import EGNNQM9PerturbRegressionTarget
from tqdm import tqdm

class CFExplainerEGNNRegressionTarget:
    def __init__(self, model, beta, device):
        self.model = model
        self.model.eval()
        self.beta = beta
        self.device = device

    def explain(
        self,
        data,
        alpha=0.1,
        tau=0.05,
        lr=1e-2,
        num_epochs=200,
        grad_clip=2.0,
        verbose=True,
        eval_every=10,
        early_stop_patience=200,
        min_epochs=50,
    ):
        data = data.to(self.device)

        # original prediction (graph-level)
        with torch.no_grad():
            y_orig = self.model(data).squeeze()  # scalar (since one graph)
        y_target = (1.0 + float(alpha)) * y_orig

        # CF model
        cf_model = EGNNQM9PerturbRegressionTarget(self.model, data).to(self.device)
        cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # freeze all weights except edge mask params
        for name, p in cf_model.named_parameters():
            p.requires_grad = (name == "p")

        cf_model.beta = self.beta

        opt = optim.Adam([cf_model.p], lr=lr)

        eval_every = max(1, int(eval_every))
        early_stop_patience = max(1, int(early_stop_patience))
        min_epochs = max(1, int(min_epochs))

        best = None
        best_loss = float("inf")
        epochs_since_best = 0

        for epoch in tqdm(
            range(num_epochs),
            desc="CF Explainer Training",
            disable=not verbose,
        ):
            cf_model.train()
            opt.zero_grad()

            y_cf_soft = cf_model(data).squeeze()

            err_soft = torch.abs(y_cf_soft.detach() - y_target)
            success = (err_soft <= tau).float()

            # Use soft-prediction success for training; hard success is checked periodically.
            loss_total, loss_pred, loss_graph = cf_model.loss_regression_target(
                y_cf_soft=y_cf_soft,
                y_target=y_target,
                success=success
            )

            loss_total.backward()
            clip_grad_norm_([cf_model.p], grad_clip)
            opt.step()

            if verbose:
                with torch.no_grad():
                    p_soft = torch.sigmoid(cf_model.p)
                    print("p_soft min/max:", p_soft.min().item(), p_soft.max().item())

            err_soft_value = err_soft.item()

            should_eval_hard = (
                epoch == 0
                or (epoch + 1) % eval_every == 0
                or epoch == (num_epochs - 1)
            )
            y_cf_hard = None
            err_hard = None
            removed = None
            success_hard = False
            if should_eval_hard:
                cf_model.eval()
                with torch.no_grad():
                    y_cf_hard, hard_edges = cf_model.forward_prediction(data, threshold=0.5)
                    y_cf_hard = y_cf_hard.squeeze()
                    err_hard = torch.abs(y_cf_hard - y_target)
                    success_hard = bool((err_hard <= tau).item())
                    removed = int(torch.sum(1.0 - hard_edges).item())
                cf_model.train()

            if verbose:
                hard_part = (
                    f"y_cf={y_cf_hard.item():.6f} err_hard={err_hard.item():.6f} removed={removed} hard_success={int(success_hard)}"
                    if should_eval_hard
                    else "hard_eval=skipped"
                )
                print(
                    f"Epoch {epoch+1:04d} | "
                    f"loss={loss_total.item():.4f} pred={loss_pred.item():.4f} graph={loss_graph.item():.4f} | "
                    f"y_orig={y_orig.item():.6f} y_target={(y_target).item():.6f} "
                    f"{hard_part} | err_soft={err_soft_value:.6f} soft_success={int(success.item())}"
                )

            if success_hard and loss_total.item() < best_loss:
                best_loss = loss_total.item()
                epochs_since_best = 0
                best = {
                    "y_orig": y_orig.detach().cpu().item(),
                    "y_target": y_target.detach().cpu().item(),
                    "y_cf": y_cf_hard.detach().cpu().item(),
                    "removed_edges": removed,
                    "hard_edge_mask_undirected": hard_edges.detach().cpu().numpy(),
                    "unique_pairs": cf_model.unique_pairs.detach().cpu().numpy(),
                    "loss_total": loss_total.item(),
                    "loss_pred": loss_pred.item(),
                    "loss_graph": loss_graph.item(),
                }
            else:
                epochs_since_best += 1

            if best is not None and (epoch + 1) >= min_epochs and epochs_since_best >= early_stop_patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1}: no better hard-success CF for {epochs_since_best} epochs."
                    )
                break

        return best
