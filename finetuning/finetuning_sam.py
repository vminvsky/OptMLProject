"""Command-line interface."""
import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
import random
from utils import create_dataloader, get_device
from optimizers.sam import SAM

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    AdamW
)

from config import (
    SARCASM_DATA_DIR,
    SENTIMENT_DATA_DIR,
    UBI_DATA_DIR,
    GPTURK_DATA_DIR
)
import sys
sys.path.append("../faithful-data-gen/src")
from data_processing import dataclass_sarcasm, dataclass_sentiment, dataclass_ubi, dataclass_gpturk


import wandb
from config import TrainerConfig

load_dotenv()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainerConfig)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# @click.command()
# @click.version_option()
@hydra.main(version_base=None, config_path="../faithful-data-gen/src/conf/trainer", config_name="trainer.yaml")
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
    
    device = get_device()

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


    train = create_dataloader(dataset.data["train"], batch_size=16)
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

    model.to(device)
    
    
    # optimizer = AdamW(model.parameters(), lr=2e-5) # Use AdamW optimizer
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3, momentum=0.9)
    scheduler = get_linear_schedule_with_warmup(optimizer, 10, 100)

    
    # Training loop
    model.train()
    for epoch in range(3): # 3 epochs
        for batch in train:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")
            loss.backward()
            optimizer.first_step(zero_grad=True)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss 
            loss.backward()
            optimizer.second_step(zero_grad=True)

    # Prepare for training
    
    print('Finished fine-tuning.')


if __name__ == "__main__":
    main()  # pragma: no cover