from dataclasses import dataclass


@dataclass
class data_config:
    image_path: str
    leader_path: str
    follower_path: str


@dataclass
class TrainerConfig:
    lr: float
    epochs: int
    device: str
    batch_size: int


@dataclass
class TrainDetails:
    name: str
    description: str
    data_config: data_config


@dataclass
class robot_details_config:
    leader_port: str
    follower_port: str


@dataclass
class TrainConfig:
    robot_details: robot_details_config
    trainer: TrainerConfig
    train_details: TrainDetails
