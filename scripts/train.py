import hydra
import wandb

from src.config.config import Config
from src.world_model.model.world_model import WorldModel
from src.utils.data_process.load_data import prepare_data_loader
from src.utils.train.trainer import Trainer
from .utils.prepare_normalizer import prepare_optimizer


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(cfg: Config):
    device = cfg.train.trainer.device
    train_dataloader, valid_dataloader, test_dataloader = prepare_data_loader(cfg)
    model = WorldModel(cfg.wm_model).to(device)
    optimizer = prepare_optimizer(cfg, model)
    trainer = Trainer(
        cfg,
        model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        optimizer,
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
