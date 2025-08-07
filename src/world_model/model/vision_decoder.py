import torch.nn as nn
import torch

from torch import Tensor
from ...config.model_config import VisionDecoderConfig
from .utils.build_mlp import build_mlp as _build_mlp
from .utils.get_conv_shape import get_conved_size


class VisionDecoder(nn.Module):
    def __init__(self, config: VisionDecoderConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = self._build_conv_layers(
            config.channels, config.kernels, config.strides, config.paddings
        )
        self.fc = _build_mlp(
            get_conved_size(
                config.obs_size,
                config.channels,
                config.kernels,
                config.strides,
                config.paddings,
            ),
            config.vision_feature_dim,
            config.mlp_hidden_dim,
            config.n_mlp_layers,
        )

    def forward(self, x) -> Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

    def _build_conv_layers(
        self,
        channels: tuple[int, ...],
        kernels: tuple[int, ...],
        strides: tuple[int, ...],
        paddings: tuple[int, ...],
    ):
        layers = []
        for i in range(len(channels)):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=channels[i - 1] if i > 0 else 3,
                    out_channels=channels[i],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
