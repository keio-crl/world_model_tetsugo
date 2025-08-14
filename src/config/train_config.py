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
class LossParametersConfig:
    kl_balance: float
    kl_beta: float
    image_recon_loss_weight: float
    follower_recon_loss_weight: float
    amplify_recon_loss: bool


@dataclass
class TrainDetails:
    project_name: str
    name: str
    description: str
    log_by_wandb: bool
    loss: LossParametersConfig


@dataclass
class RobotDetailsConfig:
    leader_port: str
    follower_port: str


@dataclass
class TrainConfig:
    trainer: TrainerConfig
    train_details: TrainDetails
