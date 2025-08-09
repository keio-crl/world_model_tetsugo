import torch
from torch.utils.data import DataLoader
from ...world_model.model.world_model import WorldModel
from ...config.config import Config


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: WorldModel,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.train.trainer.device
        self.model = model.to(self.device)

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass
