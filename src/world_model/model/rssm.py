import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from ...config.model_config import RSSMConfig
from torch import Tensor
from typing import Tuple


class RSSM(nn.Module):
    def __init__(self, config: RSSMConfig) -> None:
        super().__init__()
        self.config = config

        self.gru_cell = nn.GRUCell(
            input_size=self.config.stoch_latent_dim + self.config.action_dim,
            hidden_size=self.config.rnn_hidden_dim,
        )

        # priorはobsなし
        self.prior = nn.Sequential(
            nn.Linear(self.config.rnn_hidden_dim, self.config.rnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(
                self.config.rnn_hidden_dim // 2, self.config.stoch_latent_dim * 2
            ),  # separate to mean and std deviation
        )

        # posteriorはobs あり
        self.posterior = nn.Sequential(
            nn.Linear(
                self.config.rnn_hidden_dim + self.config.obs_dim,
                (self.config.rnn_hidden_dim + self.config.obs_dim) // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                (self.config.rnn_hidden_dim + self.config.obs_dim) // 2,
                self.config.stoch_latent_dim * 2,
            ),  # separate to mean and std deviation
        )

    def forward(
        self,
        obs: Tensor,  # (B, T, obs_dim)
        action: Tensor,  # (B, T, action_dim)
        resets: Tensor | None,  # (B, T) 0 or 1
    ) -> tuple[D.Normal, D.Normal, Tensor, Tensor]:
        B, T, _ = obs.shape

        if resets is None:
            reset_masks = None
        else:
            reset_masks = ~resets

        h_list, z_list, prior_list, posterior_list = [], [], [], []
        h = torch.zeros(B, self.config.rnn_hidden_dim, device=obs.device)
        z = torch.zeros(B, self.config.stoch_latent_dim, device=obs.device)

        for t in range(T):
            if t > 0 and reset_masks is not None:
                h = (reset_masks.unsqueeze(1) * h).float()
                z = (reset_masks.unsqueeze(1) * z).float()

            prev_action = action[:, t - 1] if t > 0 else torch.zeros_like(action[:, 0])
            rnn_input = torch.cat([z, prev_action], dim=-1)
            h = self.gru_cell(rnn_input, h)

            # Prior
            prior_params = self.prior(h)
            prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 1e-6  # Avoid zero std
            prior_dist = D.Normal(prior_mean, prior_std)

            # Posterior
            posterior_params = self.posterior(
                torch.cat([h, obs[:, t]], dim=-1)
            )  # (B, stoch_latent_dim * 2), posterior has obs information
            posterior_mean, posterior_std = torch.chunk(posterior_params, 2, dim=-1)
            posterior_std = F.softplus(posterior_std) + 1e-6  # Avoid zero std
            posterior_dist = D.Normal(posterior_mean, posterior_std)

            z = posterior_dist.rsample()  # Sample from posterior

            # Store results
            h_list.append(h)
            z_list.append(z)
            prior_list.append(prior_dist)
            posterior_list.append(posterior_dist)

        # Stack results
        h_seq = torch.stack(h_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)
        priors_list = D.Normal(
            torch.stack([d.mean for d in prior_list], dim=-1),
            torch.stack([d.scale for d in prior_list], dim=-1),
        )
        posteriors_list = D.Normal(
            torch.stack([d.mean for d in posterior_list], dim=-1),
            torch.stack([d.scale for d in posterior_list], dim=-1),
        )

        return priors_list, posteriors_list, h_seq, z_seq

    def dream(
        self,
        initial_state: Tuple[Tensor, Tensor],
        initial_action: Tensor,
        action_seq: Tensor,
    ):
        """
        推論時に観測なしで状態を予測するメソッド、priorのみを使用
        """

        h, z = initial_state
        B, T = action_seq.shape

        h_list, z_list = [], []

        for t in range(T):
            prev_action = action_seq[:, t - 1] if t > 0 else initial_action
            rnn_input = torch.cat([z, prev_action], dim=-1)
            h = self.gru_cell(rnn_input, h)

            # Prior
            prior_params = self.prior(h)
            prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 1e-6

            prior_dist = D.Normal(prior_mean, prior_std)
            z = prior_dist.rsample()  # Sample from prior

            h_list.append(h)
            z_list.append(z)

        # Stack results
        h_seq = torch.stack(h_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)

        return h_seq, z_seq
