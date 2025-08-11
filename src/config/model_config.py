from dataclasses import dataclass


@dataclass
class VisionEncoderConfig:
    """VisionEncoderの設定を管理するデータクラス"""

    obs_size: tuple[int, int]
    channels: tuple[int, ...]
    kernels: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]

    vision_feature_dim: int
    mlp_hidden_dim: int
    n_mlp_layers: int


@dataclass
class RSSMConfig:
    """RSSMの設定を管理するデータクラス"""

    vision_enocoder_config: VisionEncoderConfig
    action_dim: int
    rnn_hidden_dim: int
    stoch_latent_dim: int
    obs_dim: int
    latent_state_dim: int


@dataclass
class VisionDecoderConfig(VisionEncoderConfig):
    """VisionDecoderの設定を管理するデータクラス"""

    # VisionEncoderConfigを継承しているため、追加のフィールドは不要


@dataclass
class FollowerDecoderConfig:
    hidden_dim: int


@dataclass
class WorldModelConfig:
    """WorldModelの設定を管理するデータクラス"""

    rssm: RSSMConfig
    vision_encoder: VisionEncoderConfig
    vision_decoder: VisionDecoderConfig
    follower_decoder: FollowerDecoderConfig
    # 他のモデルやコンポーネントの設定を追加可能
