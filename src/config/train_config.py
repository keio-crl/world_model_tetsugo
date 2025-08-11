from dataclasses import dataclass


@dataclass
class data_config:
    image_path: str
    leader_path: str
    follower_path: str
    seq_length: int
    normalize_config_name: str


@dataclass
class TrainerConfig:
    lr: float
    weight_decay: float
    epochs: int
    device: str
    batch_size: int
    split_ratio: list[float]  # train, validation, test
    save_model: bool


@dataclass
class TrainDetails:
    project_name: str
    name: str
    description: str
    log_by_wandb: bool


@dataclass
class RobotDetailsConfig:
    leader_port: str
    follower_port: str


@dataclass
class TrainConfig:
    trainer: TrainerConfig
    train_details: TrainDetails
