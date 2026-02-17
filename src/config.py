from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    data_root: str = str(Path("./data").resolve())
    

    seed: int = 42

    num_workers: int = 0
    batch_size: int = 64

    image_size: int =224

    reports_dir: str = str(Path("./reports").resolve())
    figures_dir: str = str(Path("./reports/figures").resolve())

CFG = Config()