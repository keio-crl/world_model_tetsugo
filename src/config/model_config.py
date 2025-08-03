from dataclasses import dataclass


@dataclass
class RSSMConfig:
    """RSSMの設定を管理するデータクラス"""

    hidden_size: int
    state_size: int
    action_size: int
    image_size: int
    num_layers: int
