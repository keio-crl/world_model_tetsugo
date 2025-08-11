import numpy as np
import torch.distributions as D
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass


@dataclass
class LossParameters:
    origin_image: Tensor
    recon_image: Tensor
    origin_follower: Tensor
    recon_follower: Tensor
    prior_dist: D.Normal
    posterior_dist: D.Normal
    kl_balance: float = 0.5
    kl_beta: float = 1.0
    image_recon_loss_weight: float = 1.0
    follower_recon_loss_weight: float = 1.0


def kl_divergence_loss(
    prior_dist: D.Normal, posterior_dist: D.Normal, kl_balance: float
) -> Tensor:
    prior_dist_detached = D.Normal(prior_dist.mean.detach(), prior_dist.stddev.detach())
    prior_loss = D.kl_divergence(prior_dist_detached, posterior_dist)
    posterior_dist_detached = D.Normal(
        posterior_dist.mean.detach(), posterior_dist.stddev.detach()
    )
    posterior_loss = D.kl_divergence(posterior_dist_detached, prior_dist)

    kl_loss = kl_balance * posterior_loss + (1 - kl_balance) * prior_loss
    return kl_loss.mean()


def world_model_loss(params: LossParameters):
    image_recon_loss = F.mse_loss(params.recon_image, params.origin_image) * float(
        np.prod(params.origin_image.shape[1:])
    )
    follower_recon_loss = F.mse_loss(
        params.recon_follower, params.origin_follower
    ) * float(np.prod(params.origin_follower.shape[1:]))

    kl_loss = kl_divergence_loss(
        params.prior_dist, params.posterior_dist, params.kl_balance
    ) * float(np.prod(params.prior_dist.mean.shape[1:]))

    total_loss = (
        (image_recon_loss * params.image_recon_loss_weight)
        + (params.follower_recon_loss_weight * follower_recon_loss)
        + (params.kl_beta * kl_loss)
    )

    return {
        "total_loss": total_loss,
        "image_recon_loss": image_recon_loss,
        "follower_recon_loss": follower_recon_loss,
        "kl_loss": kl_loss,
    }
