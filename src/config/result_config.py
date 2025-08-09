from dataclasses import dataclass


@dataclass
class ResultConfig:
    base_path: str
    normalize_path: str
    model_dir_path: str
    evaluation_dir_path: str
    test_dir_path: str
