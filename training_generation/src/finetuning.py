"""Command-line interface."""
import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
import random
from utils import create_dataloader, get_device, StepLR
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
import time

from config import (
    SARCASM_DATA_DIR,
    UBI_DATA_DIR,
)
import sys
sys.path.append("../faithful-data-gen/src/")

from data_processing import dataclass_sarcasm, dataclass_sentiment, dataclass_ubi, dataclass_gpturk
from sklearn.metrics import f1_score, accuracy_score, precision_score


import wandb
from config import TrainerConfig

load_dotenv()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainerConfig)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def calculate_metrics(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    accuracy = accuracy_score(labels_flat, pred_flat)
    macro_f1 = f1_score(labels_flat, pred_flat, average='macro')
    precision = precision_score(labels_flat, pred_flat, average='macro')
    return accuracy, macro_f1, precision

# @click.command()
# @click.version_option()
@hydra.main(version_base=None, config_path="conf/trainer", config_name="trainer.yaml")
def main(cfg: TrainerConfig) -> None:
    SAM_ACTIVE = cfg.SAM_ACTIVE

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
    else:
        raise ValueError("Dataset not found")
    
    device = get_device()

    dataset.data["test"] = test_dataset.data["train"]
    dataset.data["base"] = base_dataset.data["train"]
    dataset.data["original_train"] = dataset.data["train"]
    dataset.data["augmented_train"] = augmented_dataset.data["train"]

    dataset.preprocess(model_name=cfg.ckpt)

    # Specify the length of train and validation set
    validation_length = 200
    if cfg.use_augmented_data:
        total_train_length = len(dataset.data["augmented_train"])
    else:
        total_train_length = len(dataset.data["train"]) - validation_length

    batch_size = cfg.batch_size
    
    num_train = 2000
    
    if cfg.use_augmented_data:
        train_dataloader = create_dataloader(dataset.data["augmented_train"].select(range(num_train)), batch_size=batch_size)
    else:
        train_dataloader = create_dataloader(dataset.data["train"].select(range(num_train)), batch_size=batch_size)
    valid = create_dataloader(dataset.data["train"], batch_size=batch_size)
    test = create_dataloader(dataset.data["test"], batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        'intfloat/e5-base',
        num_labels=len(dataset.labels)
    )

    model.to(device)
    
    
    if SAM_ACTIVE:
        lr = 2e-5
        rho = 0.003
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=lr, rho=rho, weight_decay=0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    
    
    wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"{cfg.ckpt}_optml_{SAM_ACTIVE}",
            group=f"{cfg.dataset}_{cfg.approach}"
        )
    
    # Training loop
    model.train()
    for epoch in range(4): # 3 epochs
        print(len(dataset.data["train"]))
        for batch in train_dataloader:
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")
            
            
            if SAM_ACTIVE:
                def closure():
                    loss = model(input_ids, attention_mask=attention_mask, labels=labels).loss
                    loss.backward()
                    return loss
                
                loss.backward()
                optimizer.step(closure)
                optimizer.zero_grad()
            else:
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()
        
        model.eval()
        for batch in valid:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            accuracy, macro_f1, precision =  calculate_metrics(logits, label_ids)
            wandb.log({"Validation Accuracy": accuracy, "Validation Macro F1": macro_f1, "Validation Precision": precision})
        model.train()
    # Prepare for training
    
    model.eval()
    for batch in test:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        accuracy, macro_f1, precision = calculate_metrics(logits, label_ids)
        wandb.log({"Test Accuracy": accuracy, "Test Macro F1": macro_f1, "Test Precision": precision})

    print('Finished fine-tuning.')

if __name__ == "__main__":
    main()  # pragma: no cover