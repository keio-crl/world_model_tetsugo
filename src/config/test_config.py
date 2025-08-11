from dataclasses import dataclass


@dataclass
class TestConfig:
    test_device: str
    context_length: int
