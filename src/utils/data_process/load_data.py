import os

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from src.config.config import Config
from src.utils.data_process.blosc_file_handler import BLOSCFileHandler
from src.utils.data_process.data_loader import MyDataLoader
from src.utils.data_process.dataset import MyDataset
from src.utils.normalizer.normalizer import Normalizer


def prepare_and_normalize_data(cfg: Config) -> tuple[NDArray, NDArray, NDArray]:
    images, leader, follower = map(
        BLOSCFileHandler.load,
        [
            cfg.data.image_path,
            cfg.data.leader_path,
            cfg.data.follower_path,
        ],
    )
    normalizer = Normalizer(feature_range=(-0.95, 0.95))

    images = normalizer.image_scaler(images)
    leader = normalizer.fit_transform(leader, "leader")
    follower = normalizer.fit_transform(follower, "follower")

    config_path = os.path.join(
        cfg.result.base_path,
        cfg.result.normalize_path,
        cfg.data.normalize_config_name,
    )
    os.makedirs(cfg.result.normalize_path, exist_ok=True)
    normalizer.save_config(config_path)

    return images, leader, follower


def prepare_data_loader(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    正規化されたデータを読み込み、DataLoaderを返す関数
    """
    device = cfg.train.trainer.device
    images, leader, follower = map(torch.from_numpy, prepare_and_normalize_data(cfg))

    images = images.to(device).float()
    leader = leader.to(device).float()
    follower = follower.to(device).float()

    dataset = MyDataset(cfg.train, images, leader, follower)
    data_loader = MyDataLoader(
        my_dataset=dataset,
        ratio=cfg.train.trainer.split_ratio,
        batch_size=cfg.train.trainer.batch_size,
    )

    train_loader, validation_loader, test_loader = data_loader.prepare_data()
    return train_loader, validation_loader, test_loader
