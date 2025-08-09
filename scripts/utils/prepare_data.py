import os
import torch
from torch.utils.data import DataLoader
from numpy.typing import NDArray
from src.utils.data_process.blosc_file_handler import BLOSCFileHandler
from src.config.config import Config
from src.utils.normalizer.normalizer import Normalizer
from src.utils.data_process.dataset import MyDataset
from src.utils.data_process.data_loader import MyDataLoader


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
        cfg.result.normalize_path,
        cfg.data.normalize_config_name,
    )
    os.makedirs(cfg.result.normalize_path, exist_ok=True)
    normalizer.save_config(config_path)

    return images, leader, follower


def prepare_data_loader(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """prepare_data_loader

    Args:
        cfg (Config): Configuration object containing paths and parameters for data preparation.
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train, validation ,test
    """
    device = cfg.train.trainer.device
    images, leader, follower = map(torch.from_numpy, prepare_and_normalize_data(cfg))

    images = images.to(device)
    leader = leader.to(device)
    follower = follower.to(device)

    dataset = MyDataset(cfg.train, images, leader, follower)

    data_loader = MyDataLoader(
        my_dataset=dataset,
        ratio=cfg.train.trainer.split_ratio,
        batch_size=cfg.train.trainer.batch_size,
    )

    train_loader, validation_loader, test_loader = data_loader.prepare_data()
    return train_loader, validation_loader, test_loader


def prepare_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """prepare_optimizer

    Args:
        cfg (Config): Configuration object containing optimizer parameters.
        model (torch.nn.Module): The model to optimize.
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.trainer.lr,
        weight_decay=cfg.train.trainer.weight_decay,
    )
