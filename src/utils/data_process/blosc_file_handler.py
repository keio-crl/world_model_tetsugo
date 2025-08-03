import blosc2

from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataPaths:
    """データパスのカテゴリを管理するデータクラス"""

    action: List[str] = field(default_factory=list)
    image: List[str] = field(default_factory=list)
    state: List[str] = field(default_factory=list)


@dataclass
class DataPathContainer:
    """左右のデータパスを管理するデータクラス"""

    left: DataPaths = field(default_factory=DataPaths)
    right: DataPaths = field(default_factory=DataPaths)


class BLOSCFileHandler:
    """BLOSCファイルの読み書きを行うクラス"""

    @staticmethod
    def load(file_path: str) -> NDArray:
        with open(file_path, "rb") as f:
            return blosc2.unpack_array2(f.read())

    @staticmethod
    def save(path: str, data: NDArray) -> None:
        with open(path, "wb") as f:
            f.write(blosc2.pack_array2(data))  # type: ignore
