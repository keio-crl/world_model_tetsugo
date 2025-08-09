import hydra

from src.config.config import Config
from src.world_model.model.world_model import WorldModel
from .utils.prepare_data import prepare_data_loader, prepare_optimizer
from src.utils.trainer.trainer import Trainer


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(cfg: Config):
    device = cfg.train.trainer.device
    train_dataloader, valid_dataloader, test_dataloader = prepare_data_loader(cfg)
    model = WorldModel(cfg.model).to(device)
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


if __name__ == "__main__":
    main()
