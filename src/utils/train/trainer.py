import os
import torch
import wandb

from torch import Tensor
from tqdm import tqdm
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Callable
from src.utils.train.loss_func import world_model_loss, LossParameters
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

    def train(self) -> None:
        epoch = self.cfg.train.trainer.epochs
        train_loss = []
        valid_loss = []
        test_loss = []
        loss_min = 1.0e10

        if self.cfg.train.train_details.log_by_wandb:
            wandb.init(
                project=self.cfg.train.train_details.project_name,
                name=self.cfg.train.train_details.name,
            )
            wandb.watch(self.model, log="all")

        for e in tqdm(range(epoch)):
            train_loss = train_loop(
                self.model,
                self.train_dataloader,
                self.device,
                self.loss_func,
                self.optimizer,
            )
            valid_loss = valid_loop(
                self.model, self.valid_dataloader, self.device, self.loss_func
            )
            test_loss = valid_loop(
                self.model, self.test_dataloader, self.device, self.loss_func
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
                print(f"Model saved at epoch {e} with loss {loss_min}")

    def _save_mmodel(self):
        save_path = os.path.join(
            self.cfg.result.base_path,
            self.cfg.result.model_dir_path,
        )

        save_path = os.path.join(
            save_path, f"{self.cfg.train.train_details.name}.safetensors"
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_file(self.model.state_dict(), save_path)


def train_loop(
    model: WorldModel,
    train_dataloader: DataLoader,
    decive: str,
    loss_func: Callable[[LossParameters], dict[str, Tensor]],
    optimizer: Optimizer,
):
    model.train()
    total_loss, image_recon_loss, follower_recon_loss, kl_loss = 0.0, 0.0, 0.0, 0.0

    for batch in train_dataloader:
        image, leader, follower = map(lambda x: x.to(decive), batch)
        image: Tensor
        leader: Tensor
        follower: Tensor

        reset_masks = torch.rand(image.shape[0], device=decive) < 0.0

        priors, posteriors, recon_img, recon_follower = model.forward(
            image, follower, leader, reset_masks
        )

        loss_params = LossParameters(
            image,
            recon_img,
            follower,
            recon_follower,
            priors,
            posteriors,
        )
        loss = loss_func(loss_params)

        total_loss = loss["total_loss"]
        optimizer.zero_grad()
        total_loss.backward()
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
    loss_func: Callable[[LossParameters], dict[str, Tensor]],
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

        loss_params = LossParameters(
            image,
            recon_img,
            follower,
            recon_follower,
            priors,
            posteriors,
        )
        loss = loss_func(loss_params)
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
