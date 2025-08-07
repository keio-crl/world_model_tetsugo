import torch
import torch.nn as nn


from torch import Tensor
from torch.distributions import Normal
from ...config.model_config import WorldModelConfig
from .rssm import RSSM
from .vision_encoder import VisionEncoder
from .vision_decoder import VisionDecoder


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rssm = RSSM(config.rssm)
        self.vision_encoder = VisionEncoder(config.vision_encoder)
        self.vision_decoder = VisionDecoder(config.vision_decoder)
        self.follower_decoder = nn.Sequential(
            nn.Linear(config.rssm.latent_state_dim, config.follower_decoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.follower_decoder.hidden_dim, config.rssm.action_dim),
        )

    def forward(
        self,
        images: Tensor,
        followers: Tensor,
        action_seq: Tensor,
        reset_masks: Tensor,
    ) -> tuple[Normal, Normal, Tensor, Tensor]:
        B, T, C, H, W = images.shape
        images = images.view(B * T, C, H, W)  # (B
        vision_feature = self.vision_encoder.forward(images).view(B, T, -1)

        # 画像特徴とfollower seqを結合
        obs = torch.cat([vision_feature, followers], dim=-1)

        priors, posteriors, h_seq, z_seq = self.rssm.forward(
            obs, action_seq, reset_masks
        )

        # デコーダーに入力するために画像特徴を再構築
        latent_state4decoder = torch.cat([h_seq, z_seq], dim=-1)
        recon_img = self.vision_decoder.forward(latent_state4decoder)
        recon_follower = self.follower_decoder(latent_state4decoder)

        return priors, posteriors, recon_img, recon_follower
