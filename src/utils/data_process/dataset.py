import torch

from torch.utils.data import Dataset
from typing import Tuple
from ...config.train_config import TrainConfig


class MyDataset(Dataset):
    """
    カスタムデータセットクラス。
    """

    def __init__(
        self,
        cfg: TrainConfig,
        imagedata: torch.Tensor,
        leaderdata: torch.Tensor,
        followerdata: torch.Tensor,
    ) -> None:
        super().__init__()
        device = cfg.trainer.device
        self.imagedata = imagedata.to(device)
        self.leaderdata = leaderdata.to(device)
        self.followerdata = followerdata.to(device)
        self.device = device
        self.cfg = cfg

    def __len__(self) -> int:
        # データセットの長さを画像データの長さに基づいて返す
        return len(self.imagedata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        image = self.imagedata[index]
        leader = self.leaderdata[index]
        follower = self.followerdata[index]
        return image, leader, follower

    def __repr__(self) -> str:
        return f"MyDataset(imagedata={self.imagedata.shape}, leaderdata={self.leaderdata.shape}, followerdata={self.followerdata.shape})"
