import os
from safetensors.torch import load_file
from src.config.config import Config
from src.world_model.model.world_model import WorldModel


def load_wm_model(cfg: Config) -> WorldModel:
    model = WorldModel(cfg.wm_model)
    model_path = os.path.join(
        cfg.result.base_path,
        cfg.result.model_dir_path,
        cfg.train.train_details.name + ".safetensors",
    )
    assert os.path.exists(model_path), f"Model file {model_path} does not exist."
    model.load_state_dict(load_file(model_path))
    model.to(cfg.train.trainer.device)
    model.eval()

    return model
