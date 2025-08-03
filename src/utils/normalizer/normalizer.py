import torch
import numpy as np
import json
import os
from typing import Callable, Tuple, Dict, Optional, Union


class Normalizer:
    def __init__(
        self, feature_range: Tuple[float, float], config_path: Optional[str] = None
    ) -> None:
        self.feature_range = feature_range
        self.min_vals: Dict[str, Union[float, np.ndarray]] = {}
        self.max_vals: Dict[str, Union[float, np.ndarray]] = {}
        self.config_path = config_path
        self.image_scaler: Callable[[np.ndarray], np.ndarray] = lambda x: x / 255.0
        self.image_scaler_inv: Callable[[torch.Tensor], torch.Tensor] = (
            lambda x: x * 255.0
        )

        # configから読み込む場合
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def fit(self, data: np.ndarray, feature_name: str) -> "Normalizer":
        """データからmin/maxを計算して保存"""
        assert feature_name == "leader" or feature_name == "follower", (
            "feature_name must be 'leader' or 'follower'."
        )
        self.min_vals[feature_name] = data.min(axis=(0, 1))
        self.max_vals[feature_name] = data.max(axis=(0, 1))
        return self

    def transform(self, data: np.ndarray, feature_name: str) -> np.ndarray:
        """データを正規化"""
        assert feature_name == "leader" or feature_name == "follower", (
            "feature_name must be 'leader' or 'follower'."
        )

        if feature_name not in self.min_vals or feature_name not in self.max_vals:
            raise ValueError(
                f"Feature '{feature_name}' is not fitted yet. Call fit() first."
            )

        min_val = self.min_vals[feature_name]
        max_val = self.max_vals[feature_name]

        a, b = self.feature_range
        return ((data - min_val) / (max_val - min_val)) * (b - a) + a

    def fit_transform(self, data: np.ndarray, feature_name: str) -> np.ndarray:
        """fitとtransformを一度に実行"""
        assert feature_name == "leader" or feature_name == "follower", (
            "feature_name must be 'leader' or 'follower'."
        )
        self.fit(data, feature_name)
        return self.transform(data, feature_name)

    def inverse_transform(self, data: np.ndarray, feature_name: str) -> np.ndarray:
        """正規化されたデータを元のスケールに戻す"""
        assert feature_name == "leader" or feature_name == "follower", (
            "feature_name must be 'leader' or 'follower'."
        )
        if feature_name not in self.min_vals or feature_name not in self.max_vals:
            raise ValueError(f"Feature '{feature_name}' is not fitted yet.")

        min_val = self.min_vals[feature_name]
        max_val = self.max_vals[feature_name]

        a, b = self.feature_range
        return ((data - a) / (b - a)) * (max_val - min_val) + min_val

    def save_config(self, config_path: Optional[str] = None) -> None:
        """min/maxの値をJSONファイルとして保存"""
        path = config_path or self.config_path
        if not path:
            raise ValueError("No config path provided.")

        config = {
            "feature_range": self.feature_range,
            "min_vals": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.min_vals.items()
            },
            "max_vals": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.max_vals.items()
            },
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self, config_path: Optional[str] = None) -> None:
        """JSONファイルからmin/maxの値を読み込む"""
        path = config_path or self.config_path
        if not path:
            raise ValueError("No config path provided.")

        with open(path, "r") as f:
            config = json.load(f)

        self.feature_range = tuple(config["feature_range"])
        self.min_vals = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in config["min_vals"].items()
        }
        self.max_vals = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in config["max_vals"].items()
        }
