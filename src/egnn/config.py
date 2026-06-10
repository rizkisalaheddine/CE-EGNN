from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_root: str = "data/QM9"

    batch_size: int = 96
    num_epochs: int = 1000

    base_lr: float = 5e-4
    lr_homo_lumo_gap: float = 1e-3

    weight_decay: float = 1e-16

    hidden_dim: int = 128
    depth: int = 7

    use_cosine_lr: bool = True

    coord_updates: bool = False

    property_name: str = "mu"

    patience: int = 50
    min_delta: float = 1e-4
