"""Command-line interface."""
import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
import random

from data_processing import dataclass_sarcasm, dataclass_sentiment, dataclass_ubi, dataclass_gpturk

from config import (
    SARCASM_DATA_DIR,
    SENTIMENT_DATA_DIR,
    UBI_DATA_DIR,
    GPTURK_DATA_DIR
)

import wandb
from config import TrainerConfig

from finetuning.trainers import ExperimentTrainer

load_dotenv()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainerConfig)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# @click.command()
# @click.version_option()
@hydra.main(version_base=None, config_path="conf/trainer", config_name="trainer.yaml")
def main(cfg: TrainerConfig) -> None:
    if cfg.dataset == "sarcasm":
        dataset = dataclass_sarcasm.SarcasmDataset(
            SARCASM_DATA_DIR / "train.json"
        )
        test_dataset = dataclass_sarcasm.SarcasmDataset(
            SARCASM_DATA_DIR / "test.json"
        )
        base_dataset = dataclass_sarcasm.SarcasmDataset(
            SARCASM_DATA_DIR / "base.json"
        )
        augmented_dataset = dataclass_sarcasm.SarcasmDataset(
            SARCASM_DATA_DIR
            / f"{cfg.approach}_{cfg.augmentation_model}.json",
            is_augmented=True,
        )
    elif cfg.dataset == "sentiment":
        dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR / "train.json"
        )
        test_dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR / "test.json"
        )
        base_dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR / "base.json"
        )
        augmented_dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR
            / f"{cfg.approach}_{cfg.augmentation_model}.json",
            is_augmented=True,
        )
    elif cfg.dataset == "ubi":
        dataset = dataclass_ubi.UBIDataset(
            UBI_DATA_DIR / "train.json"
        )
        test_dataset = dataclass_ubi.UBIDataset(
            UBI_DATA_DIR / "test.json"
        )
        base_dataset = dataclass_ubi.UBIDataset(
             UBI_DATA_DIR / "train.json"
        )
        augmented_dataset = dataclass_ubi.UBIDataset(
            UBI_DATA_DIR / "train.json",
        )
    elif cfg.dataset == "gpturk_inductive":
        dataset = dataclass_gpturk.GPTurkDataset(
            GPTURK_DATA_DIR / "train_inductive.json"
        )
        test_dataset = dataclass_gpturk.GPTurkDataset(
            GPTURK_DATA_DIR / "test_inductive.json"
        )
        base_dataset = dataclass_gpturk.GPTurkDataset(
             GPTURK_DATA_DIR / "train_inductive.json"
        )
        augmented_dataset = dataclass_gpturk.GPTurkDataset(
            GPTURK_DATA_DIR / "train_inductive.json",
        )
    elif cfg.dataset == "gpturk_transductive":
        dataset = dataclass_gpturk.GPTurkDataset(
            GPTURK_DATA_DIR / "train_transductive.json"
        )
        test_dataset = dataclass_gpturk.GPTurkDataset(
            GPTURK_DATA_DIR / "test_transductive.json"
        )
        base_dataset = dataclass_gpturk.GPTurkDataset(
             GPTURK_DATA_DIR / "train_transductive.json"
        )
        augmented_dataset = dataclass_gpturk.GPTurkDataset(
            GPTURK_DATA_DIR / "train_transductive.json",
        )
    else:
        raise ValueError("Dataset not found")

    dataset.data["test"] = test_dataset.data["train"]
    dataset.data["base"] = base_dataset.data["train"]
    dataset.data["original_train"] = dataset.data["train"]
    dataset.data["augmented_train"] = augmented_dataset.data["train"]

    dataset.preprocess(model_name=cfg.ckpt)

    # Specify the length of train and validation set
    validation_length = 50
    if cfg.use_augmented_data:
        total_train_length = len(dataset.data["augmented_train"])
    else:
        total_train_length = len(dataset.data["train"]) - validation_length

    # generate list of indices to slice from
    # indices = list(range(0, total_train_length, 500)) + [total_train_length]
    
    indices = list(range(0, total_train_length, 500)) # + [total_train_length]

    # Select only indices with value 5000 or less
    indices = [idx for idx in indices if idx <= 5000]

    for idx in indices:
        if cfg.use_augmented_data:
            if idx == 0:
                continue
            dataset.exp_datasize_split_aug(idx, validation_length)
        else:
            dataset.exp_datasize_split(idx, validation_length, cfg.use_augmented_data)

        model = ExperimentTrainer(data=dataset, config=cfg)

        model.train()

        model.test()

        wandb.finish()


if __name__ == "__main__":
    main()  # pragma: no cover