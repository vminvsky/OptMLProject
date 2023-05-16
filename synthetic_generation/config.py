from pathlib import Path
from dataclasses import dataclass

# Defining paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

ANALYSE_TAL_DATA_DIR = DATA_DIR / "analyse-tal"

SENTIMENT_DATA_DIR = DATA_DIR / "sentiment"

TEN_DIM_DATA_DIR = DATA_DIR / "ten-dim"

SRC_DIR = ROOT_DIR / "src"

WORKER_VS_GPT_DIR = SRC_DIR / "worker_vs_gpt"


@dataclass
class TrainerConfig:
    """Trainer config class."""

    ckpt: str
    dataset: str
    use_augmented_data: bool
    sampling: str
    augmentation_model: str
    wandb_project: str
    wandb_entity: str
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float


@dataclass
class PromptConfig:
    """Config for prompting classification."""

    model: str
    dataset: str
    wandb_project: str
    wandb_entity: str


@dataclass
class SimilarityConfig:
    """Config for prompting classification."""

    model: str
    dataset: str
    augmentation: str
    use_augmented_data: bool


class AugmentConfig:
    """Config for prompting augmentation."""

    model: str
    dataset: str
    sampling: str
