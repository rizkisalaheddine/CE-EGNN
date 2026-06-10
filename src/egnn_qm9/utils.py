# src/egnn_qm9/utils.py

import torch
import json
import torch
from src.egnn_qm9.model import EGNNQM9Model

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained_model(property: str, device: torch.device):
    cfg_path  = f"models/{property}/config.json"
    ckpt_path = f"models/{property}/best_model_state.pt"

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    state = torch.load(ckpt_path, map_location=device)

    return cfg, state