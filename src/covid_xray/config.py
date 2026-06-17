from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Default project paths and training settings."""

    data_dir: Path = Path("../covid-19/DataSet/Data")
    output_dir: Path = Path("artifacts")
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.2
    seed: int = 42
    learning_rate: float = 1e-4
    epochs: int = 20

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"
