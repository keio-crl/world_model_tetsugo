import numpy as np
import torch.distributions as D
import torch.nn.functional as F

from torch import Tensor
from dataclasses import dataclass
from rich import print


@dataclass
class LossParameters:
    origin_image: Tensor
    recon_image: Tensor
    origin_follower: Tensor
    recon_follower: Tensor
    prior_dist: D.Normal
    posterior_dist: D.Normal
    kl_balance: float
    kl_beta: float
    image_recon_loss_weight: float
    follower_recon_loss_weight: float


# 修正後のコード
def kl_divergence_loss(
    prior_dist: D.Normal, posterior_dist: D.Normal, kl_balance: float
) -> Tensor:
    """
    KLダイバージェンスを計算します（KLバランシング使用）。
    - posterior_loss: 表現モデル（事後分布）を更新するための損失。
    - prior_loss: ダイナミクスモデル（事前分布）を更新するための損失。
    """
    # 1. posterior を更新するための損失 (Representation Loss)
    #    KL[posterior || sg(prior)]
    #    prior_dist の勾配は .detach()
    prior_dist_detached = D.Normal(prior_dist.mean.detach(), prior_dist.stddev.detach())
    posterior_loss = D.kl_divergence(posterior_dist, prior_dist_detached)

    # 2. prior を更新するための損失 (Dynamics Loss)
    #    KL[sg(posterior) || prior] を計算。
    #    posterior_dist の勾配は .detach() で停止
    posterior_dist_detached = D.Normal(
        posterior_dist.mean.detach(), posterior_dist.stddev.detach()
    )
    prior_loss = D.kl_divergence(posterior_dist_detached, prior_dist)

    # 3. KLバランシング
    #    kl_balance (alpha) の重みで posterior_loss を、
    #    (1 - kl_balance) の重みで prior_loss を適用
    kl_loss = kl_balance * posterior_loss + (1 - kl_balance) * prior_loss

    return kl_loss.mean()


def world_model_loss(params: LossParameters, amplify_recon_loss: bool):
    # world_model_loss
    if amplify_recon_loss:
        image_recon_loss = F.mse_loss(params.recon_image, params.origin_image) * float(
            np.prod(params.origin_image.shape[-3:])
        )
        follower_recon_loss = F.mse_loss(
            params.recon_follower, params.origin_follower
        ) * float(np.prod(params.origin_follower.shape[-1]))

        kl_loss = kl_divergence_loss(
            params.prior_dist, params.posterior_dist, params.kl_balance
        ) * float(np.prod(params.prior_dist.mean.shape[-1]))
    else:
        image_recon_loss = F.mse_loss(params.recon_image, params.origin_image)
        follower_recon_loss = F.mse_loss(params.recon_follower, params.origin_follower)
        kl_loss = kl_divergence_loss(
            params.prior_dist, params.posterior_dist, params.kl_balance
        )

    # 損失の合計を計算
    total_loss = (
        (image_recon_loss * params.image_recon_loss_weight)
        + (params.follower_recon_loss_weight * follower_recon_loss)
        + (params.kl_beta * kl_loss)
    )

    loss = {
        "total_loss": total_loss,
        "image_recon_loss": image_recon_loss,
        "follower_recon_loss": follower_recon_loss,
        "kl_loss": kl_loss,
    }

    print("Losses:", loss)  # デバッグ用出力

    return loss
