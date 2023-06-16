from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import datasets
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from data_processing.dataclass import DataClassWorkerVsGPT

torch.manual_seed(42)

from config import SARCASM_DATA_DIR


class UBIDataset(DataClassWorkerVsGPT):
    def __init__(
            self,
            path: Union[Path, None],
            labels=[
                "a",
                "n",
                "s"
            ],
            is_augmented: bool = False,
    ) -> None:
        super().__init__(path, is_augmented)
        self.labels: List[str] = labels
        self.is_augmented: bool = is_augmented

    def preprocess(self, model_name: str) -> None:
        # Convert labels to int
        self.data = self.data.rename_column("labels", "labels_og")
        self.data = self.data.map(
            lambda x: {"labels": self._label_preprocessing(x["labels_og"])},
        )

        # Define tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # tokenize the text
        if self.is_augmented:
            self.text_column = "augmented_text"
        else:
            self.text_column = "text"

        self.data = self.data.map(
            lambda x: tokenizer(x[self.text_column], truncation=True, padding="max_length", max_length=128),
            batched=True,
        )
        # convert the text to tensor
        self.data = self.data.map(
            lambda x: {"input_ids": x["input_ids"]},
            batched=True,
        )

        # convert the attention mask to tensor
        self.data = self.data.map(
            lambda x: {"attention_mask": x["attention_mask"]},
            batched=True,
        )

        # Format columns to torch tensors
        self.data.set_format("torch")

        # Format labels column to torch tensor with dtype float32
        self.data = self.data.map(
            lambda x: {"float_labels": x["labels"].to(torch.float)},
            remove_columns=["labels"],
        ).rename_column("float_labels", "labels")

        # Cast target to ClassLabel
        self.data = self.data.cast_column(
            "labels_og", datasets.ClassLabel(names=self.labels)
        )

if __name__ == "__main__":
    print("Hello world!")

    path = SENTIMENT_DATA_DIR / "train.json"

    data = SentimentDataset(path)

    data.preprocess(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    data.make_static_baseset()

    # Specify the length of train and validation set
    baseset_length = 500
    validation_length = 200
    total_train_length = len(data.data["train"]) - validation_length - baseset_length

    # generate list of indices jumping by 500, and the last index is the length of the dataset
    indices = list(range(0, total_train_length, 500)) + [total_train_length]

    for idx in indices:
        data.exp_datasize_split(idx, validation_length)
        print(data.data)
        print("------------")

    processed_data = data.get_data()

    a = 1
