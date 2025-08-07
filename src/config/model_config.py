from dataclasses import dataclass, field


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
    obs_dim: int = field(init=False)
    rnn_hidden_dim: int
    stoch_latent_dim: int
    latent_state_dim: int = field(init=False)

    def __post_init__(self):
        # 初期化あとにobs_dimをaction_dimとvision_encoderのmlp_hidden_dimを使って計算する
        if self.vision_enocoder_config.n_mlp_layers > 0:
            self.obs_dim = self.vision_enocoder_config.mlp_hidden_dim + self.action_dim
        else:
            self.obs_dim = (
                self.vision_enocoder_config.vision_feature_dim + self.action_dim
            )

        # latent_state_dimの計算
        self.latent_state_dim = self.rnn_hidden_dim + self.stoch_latent_dim


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
