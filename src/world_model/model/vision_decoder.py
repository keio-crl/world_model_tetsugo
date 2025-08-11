import torch.nn as nn
from torch import Tensor
from ...config.model_config import WorldModelConfig
from .common.build_mlp import build_mlp as _build_mlp


class VisionDecoder(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder_config = config.vision_encoder
        self.decoder_config = config.vision_decoder

        # 1.【修正】Encoder側のパラメータで最初の特徴マップサイズを計算
        self.conv_in_h, self.conv_in_w = self._get_initial_feature_hw(
            self.encoder_config.obs_size,
            self.encoder_config.kernels,
            self.encoder_config.strides,
            self.encoder_config.paddings,
        )
        # Decoderの最初の入力チャネル数は、Encoderの最後のチャネル数
        self.conv_in_c = self.encoder_config.channels[-1]
        self.conv_in_dim = self.conv_in_c * self.conv_in_h * self.conv_in_w

        # MLPで (B*T, D) -> (B*T, C*H*W)
        self.fc = _build_mlp(
            self.encoder_config.vision_feature_dim,  # 入力512次元
            self.conv_in_dim,
            self.encoder_config.mlp_hidden_dim,
            self.encoder_config.n_mlp_layers,
        )

        # 2.【修正】`output_paddings` を渡してデコーダーを構築
        self.decoder = self._build_conv_layers(
            self.decoder_config.channels,  # 例: (128, 64, 32)
            self.decoder_config.kernels,
            self.decoder_config.strides,
            self.decoder_config.paddings,
        )

    def forward(self, x: Tensor) -> Tensor:
        # 入力形状: (B, T, D) e.g., (16, 60, 512)
        B, T, D = x.shape

        # (B, T, D) -> (B*T, D) に変形して一度に処理
        x = x.reshape(B * T, D)

        # 全結合層で特徴マップの次元に変換
        x = self.fc(x)

        # (B*T, C*H*W) -> (B*T, C, H, W) に復元
        x = x.view(B * T, self.conv_in_c, self.conv_in_h, self.conv_in_w)
        # 転置畳み込みで画像を再構成
        recon_x: Tensor = self.decoder(x)
        recon_x = recon_x.view(B, T, *recon_x.shape[1:])

        return recon_x

    def _get_initial_feature_hw(self, obs_size, kernels, strides, paddings):
        h, w = obs_size
        for k, s, p in zip(kernels, strides, paddings):
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
        return h, w

    def _build_conv_layers(self, channels, kernels, strides, paddings):
        layers = []
        for i in range(len(channels)):
            in_c = channels[i]
            out_c = 3 if i == len(channels) - 1 else channels[i + 1]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=1,
                )
            )
            if i < len(channels) - 1:
                layers.append(nn.ReLU())

        layers.append(nn.Tanh())
        return nn.Sequential(*layers)
