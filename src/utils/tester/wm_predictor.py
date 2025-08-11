import torch
from torch import Tensor
from ...world_model.model.world_model import WorldModel
from ...config.config import Config
from ..data_process.load_data import prepare_data_loader


class WMPredictor:
    def __init__(self, cfg: Config, wm_model: WorldModel) -> None:
        self.cfg = cfg
        self.wm_model = wm_model

    @torch.no_grad()
    def predict(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        images, leader, follower = self._prepare_data(self.cfg)
        contex_len = self.cfg.test.context_length

        # 予測のためのコンテキストデータを取得
        context_images = images[:, :contex_len]
        context_leader = leader[:, :contex_len]
        context_follower = follower[:, :contex_len]

        B, T, C, H, W = context_images.shape
        context_images = context_images.reshape(B * T, C, H, W)
        vision_feat = self.wm_model.vision_encoder.forward(context_images)
        vision_feat = vision_feat.view(B, T, -1)

        obs = torch.cat([vision_feat, context_follower], dim=-1)

        _, _, context_h_seq, context_z_seq = self.wm_model.rssm.forward(
            obs, context_leader, None
        )

        last_h = context_h_seq[:, -1]
        last_z = context_z_seq[:, -1]
        inital_state = (last_h, last_z)
        inital_action = context_leader[:, -1]

        leader2dream = leader[:, contex_len:]

        dreamed_h_seq, dreamed_z_seq = self.wm_model.rssm.dream(
            inital_state, inital_action, leader2dream
        )

        context_latent_state = torch.cat([context_h_seq, context_z_seq], dim=-1)
        dreamed_latent_state = torch.cat([dreamed_h_seq, dreamed_z_seq], dim=-1)

        # 再構築
        recon_image = self.wm_model.vision_decoder.forward(dreamed_latent_state)
        recon_follower = self.wm_model.follower_decoder.forward(dreamed_latent_state)

        context_recon_images = self.wm_model.vision_decoder.forward(
            context_latent_state
        )
        context_recon_follower = self.wm_model.follower_decoder.forward(
            context_latent_state
        )

        return context_recon_images, context_recon_follower, recon_image, recon_follower

    def _prepare_data(self, cfg: Config) -> tuple[Tensor, Tensor, Tensor]:
        _, _, test_dataloader = prepare_data_loader(cfg)
        data: tuple[Tensor, Tensor, Tensor] = next(iter(test_dataloader))
        images, leader, follower = data
        device = getattr(self.wm_model, "device", None)

        images = images.to(device)
        leader = leader.to(device)
        follower = follower.to(device)

        return images, leader, follower
