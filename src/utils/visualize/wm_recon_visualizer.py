import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import print

from ...config.config import Config
from ..tester.wm_predictor import WMPredictions


class WMReconVisualizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.save_dir = os.path.join(
            self.cfg.result.base_path,
            self.cfg.result.evaluation_dir_path,
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def visualize_image(self, prediction: WMPredictions) -> None:
        origin_images = prediction["original_images"].squeeze(0)
        recon_images = prediction["recon_image"]
        context_recon_images = prediction["context_recon_images"]
        full_recon_images = torch.cat(
            [context_recon_images, recon_images], dim=1
        ).squeeze(0)

        print(f"Original images shape: {origin_images.shape}")
        print(f"Reconstructed images shape: {full_recon_images.shape}")

        # GIF動画を作成
        self.create_comparison_gif_with_labels(
            original_images=origin_images,
            recon_images=full_recon_images,
            fps=10,
        )

    def create_comparison_gif_with_labels(
        self, original_images: torch.Tensor, recon_images: torch.Tensor, fps: int = 10
    ) -> None:
        """
        ラベル付きの比較GIFを作成（matplotlibを使用）

        Args:
            original_images: (T, C, H, W) - オリジナル画像シーケンス
            recon_images: (T, C, H, W) - 再構築画像シーケンス
            fps: フレームレート
        """
        # GPUからCPUに移動し、numpy配列に変換
        original_np = original_images.detach().cpu().numpy()
        recon_np = recon_images.detach().cpu().numpy()

        # 値の範囲を[0, 1]に正規化
        original_np = np.clip(original_np, 0, 1)
        recon_np = np.clip(recon_np, 0, 1)

        # (T, C, H, W) -> (T, H, W, C) に変換
        original_np = original_np.transpose(0, 2, 3, 1)
        recon_np = recon_np.transpose(0, 2, 3, 1)

        num_frames = original_np.shape[0]

        # Figure作成
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].set_title("Original")
        axes[1].set_title("Reconstructed")

        # 初期フレーム表示
        im1 = axes[0].imshow(original_np[0])
        im2 = axes[1].imshow(recon_np[0])

        # 軸を非表示
        for ax in axes:
            ax.axis("off")
        # レイアウト調整
        plt.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.01, wspace=0.02)
        fig.tight_layout(pad=0.5)

        def animate(frame_idx):
            """アニメーション更新関数"""
            im1.set_array(original_np[frame_idx])
            im2.set_array(recon_np[frame_idx])
            fig.suptitle(f"Frame {frame_idx + 1}/{num_frames}")
            return [im1, im2]

        # アニメーション作成
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=num_frames,
            interval=int(1000 / fps),
            blit=True,
            repeat=True,
        )

        # GIFとして保存
        gif_path = os.path.join(
            self.save_dir,
            self.cfg.result.evaluation_dir_path,
            self.cfg.train.train_details.name,
            self.cfg.train.train_details.description,
            "recon_image.gif",
        )
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        gif_path = get_unique_filepath(gif_path)  # ←ここを追加
        anim.save(gif_path, writer="pillow", fps=fps)

        plt.close(fig)
        print(f"[green]Labeled GIF saved to: {gif_path}[/green]")

    def create_side_by_side_frames(
        self, original_images: torch.Tensor, recon_images: torch.Tensor
    ) -> None:
        """
        各フレームを個別に保存（デバッグ用）
        """
        original_np = original_images.detach().cpu().numpy()
        recon_np = recon_images.detach().cpu().numpy()

        original_np = np.clip(original_np, 0, 1)
        recon_np = np.clip(recon_np, 0, 1)

        original_np = original_np.transpose(0, 2, 3, 1)
        recon_np = recon_np.transpose(0, 2, 3, 1)

        frames_dir = os.path.join(self.save_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i in range(original_np.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(original_np[i])
            axes[0].set_title(f"Original - Frame {i + 1}")
            axes[0].axis("off")

            axes[1].imshow(recon_np[i])
            axes[1].set_title(f"Reconstructed - Frame {i + 1}")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(frames_dir, f"frame_{i:03d}.png"))
            plt.close()

        print(f"[green]Individual frames saved to: {frames_dir}[/green]")


def get_unique_filepath(filepath: str) -> str:
    """
    ファイルパスが既に存在する場合、末尾に連番を付与してユニークなパスを返す
    例: foo.gif, foo_1.gif, foo_2.gif, ...
    """
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath
    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{ext}"
        counter += 1
    return new_filepath
