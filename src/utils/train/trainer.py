import os
from typing import Callable

import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from ...config.config import Config
from ...utils.train.loss_func import LossParameters, world_model_loss
from ...utils.train.save_model import ModelSaver
from ...world_model.model.world_model import WorldModel


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: WorldModel,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: Optimizer,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.train.trainer.device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_func = world_model_loss
        self.model_saver = ModelSaver()

    def train(self) -> None:
        epoch = self.cfg.train.trainer.epochs
        train_loss = []
        valid_loss = []
        test_loss = []
        loss_min = 1.0e10

        if self.cfg.train.train_details.log_by_wandb:
            # omegaconfを使用してConfigを辞書形式に変換し、型を明示的にキャスト
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
            if not isinstance(config_dict, dict):
                raise ValueError("Config could not be converted to a dictionary.")

            # 型を明示的にキャスト
            config_dict = {str(k): v for k, v in config_dict.items()}

            wandb.init(
                project=self.cfg.train.train_details.project_name,
                name=self.cfg.train.train_details.name,
                notes=self.cfg.train.train_details.description,
                config=config_dict,  # 辞書形式のConfigを渡す
            )
            wandb.watch(self.model, log="all")

        for e in tqdm(range(epoch)):
            train_loss = train_loop(
                self.model,
                self.train_dataloader,
                self.device,
                self.loss_func,
                self.optimizer,
                self.cfg,
            )
            valid_loss = valid_loop(
                self.model, self.valid_dataloader, self.device, self.loss_func, self.cfg
            )
            test_loss = valid_loop(
                self.model, self.test_dataloader, self.device, self.loss_func, self.cfg
            )

            if self.cfg.train.train_details.log_by_wandb:
                wandb.log(
                    {
                        "epoch": e,
                        "train_loss/total_loss": train_loss["total_loss"],
                        "train_loss/image_recon_loss": train_loss["image_recon_loss"],
                        "train_loss/follower_recon_loss": train_loss[
                            "follower_recon_loss"
                        ],
                        "train_loss/kl_loss": train_loss["kl_loss"],
                        "valid_loss/total_loss": valid_loss["total_loss"],
                        "valid_loss/image_recon_loss": valid_loss["image_recon_loss"],
                        "valid_loss/follower_recon_loss": valid_loss[
                            "follower_recon_loss"
                        ],
                        "valid_loss/kl_loss": valid_loss["kl_loss"],
                        "test_loss/total_loss": test_loss["total_loss"],
                        "test_loss/image_recon_loss": test_loss["image_recon_loss"],
                        "test_loss/follower_recon_loss": test_loss[
                            "follower_recon_loss"
                        ],
                        "test_loss/kl_loss": test_loss["kl_loss"],
                    }
                )
            if (
                valid_loss["total_loss"] < loss_min
                and self.cfg.train.trainer.save_model
            ):
                loss_min = valid_loss["total_loss"]
                self._save_mmodel()

    def _save_mmodel(self):
        save_path = os.path.join(
            self.cfg.result.base_path,
            self.cfg.result.model_dir_path,
        )

        save_path = os.path.join(
            save_path, f"{self.cfg.train.train_details.name}.safetensors"
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.model_saver.add(self.model.state_dict(), save_path)


def train_loop(
    model: WorldModel,
    train_dataloader: DataLoader,
    decive: str,
    loss_func: Callable[[LossParameters, bool], dict[str, Tensor]],
    optimizer: Optimizer,
    config: Config,
):
    model.train()
    total_loss, image_recon_loss, follower_recon_loss, kl_loss = 0.0, 0.0, 0.0, 0.0

    for batch in train_dataloader:
        image, leader, follower = map(lambda x: x.to(decive), batch)
        image: Tensor
        leader: Tensor
        follower: Tensor

        reset_masks = torch.rand(image.shape[0], device=decive) < 0.5

        priors, posteriors, recon_img, recon_follower = model.forward(
            image, follower, leader, reset_masks
        )

        loss_config = config.train.train_details.loss

        loss_params = LossParameters(
            image,
            recon_img,
            follower,
            recon_follower,
            priors,
            posteriors,
            loss_config.kl_balance,
            loss_config.kl_beta,
            loss_config.image_recon_loss_weight,
            loss_config.follower_recon_loss_weight,
        )

        loss = loss_func(
            loss_params, config.train.train_details.loss.amplify_recon_loss
        )

        train_total_loss = loss["total_loss"]
        optimizer.zero_grad()
        train_total_loss.backward()

        optimizer.step()

        total_loss += loss["total_loss"].item()
        image_recon_loss += loss["image_recon_loss"].item()
        follower_recon_loss += loss["follower_recon_loss"].item()
        kl_loss += loss["kl_loss"].item()

    average_loss = total_loss / len(train_dataloader)
    average_image_recon_loss = image_recon_loss / len(train_dataloader)
    average_follower_recon_loss = follower_recon_loss / len(train_dataloader)
    average_kl_loss = kl_loss / len(train_dataloader)

    return {
        "total_loss": average_loss,
        "image_recon_loss": average_image_recon_loss,
        "follower_recon_loss": average_follower_recon_loss,
        "kl_loss": average_kl_loss,
    }


@torch.no_grad()
def valid_loop(
    model: WorldModel,
    dataloader: DataLoader,
    decive: str,
    loss_func: Callable[[LossParameters, bool], dict[str, Tensor]],
    config: Config,
):
    model.eval()
    total_loss, image_recon_loss, follower_recon_loss, kl_loss = 0.0, 0.0, 0.0, 0.0

    for batch in dataloader:
        image, leader, follower = map(lambda x: x.to(decive), batch)
        image: Tensor
        leader: Tensor
        follower: Tensor

        priors, posteriors, recon_img, recon_follower = model.forward(
            image,
            follower,
            leader,
        )

        loss_config = config.train.train_details.loss
        loss_params = LossParameters(
            image,
            recon_img,
            follower,
            recon_follower,
            priors,
            posteriors,
            loss_config.kl_balance,
            loss_config.kl_beta,
            loss_config.image_recon_loss_weight,
            loss_config.follower_recon_loss_weight,
        )
        loss = loss_func(
            loss_params, config.train.train_details.loss.amplify_recon_loss
        )
        total_loss += loss["total_loss"].item()
        image_recon_loss += loss["image_recon_loss"].item()
        follower_recon_loss += loss["follower_recon_loss"].item()
        kl_loss += loss["kl_loss"].item()

    average_loss = total_loss / len(dataloader)
    average_image_recon_loss = image_recon_loss / len(dataloader)
    average_follower_recon_loss = follower_recon_loss / len(dataloader)
    average_kl_loss = kl_loss / len(dataloader)

    return {
        "total_loss": average_loss,
        "image_recon_loss": average_image_recon_loss,
        "follower_recon_loss": average_follower_recon_loss,
        "kl_loss": average_kl_loss,
    }
