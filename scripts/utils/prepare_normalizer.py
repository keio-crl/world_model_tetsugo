import torch
from src.config.config import Config


def prepare_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    optimizerを準備する関数
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.trainer.lr,
        weight_decay=cfg.train.trainer.weight_decay,
    )
