import json

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(property: str, device: torch.device):
    cfg_path = f"models/{property}/config.json"
    ckpt_path = f"models/{property}/best_model_state.pt"

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    state = torch.load(ckpt_path, map_location=device)

    return cfg, state
