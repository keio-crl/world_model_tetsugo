from dataclasses import dataclass
from typing import Literal
from .model_config import WorldModelConfig
from .train_config import TrainConfig, RobotDetailsConfig, data_config
from .result_config import ResultConfig


@dataclass
class Config:
    mode: Literal["train", "eval", "test"]
    model: WorldModelConfig
    train: TrainConfig
    robot_details: RobotDetailsConfig
    data: data_config
    result: ResultConfig
