from typing import Dict, Iterable, List, Union
import torch
from pathlib import Path
from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size, shuffle=True):
    """
    Create a DataLoader instance from a dataset.

    Parameters:
    dataset: the dataset to load
    batch_size (int): the number of samples per batch
    shuffle (bool, optional): whether to shuffle the data. Default is True.

    Returns:
    DataLoader: a DataLoader instance
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class StepLR:
    """
    Step Learning Rate scheduler.
    Adjusts the learning rate according to the number of epochs passed.
    """

    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        # Initializes the StepLR class.
        
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        # Adjusts the learning rate according to the current epoch.
        
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3
        # Apply the learning rate to all parameter groups in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        # Returns the current learning rate.

        return self.optimizer.param_groups[0]["lr"]


def parse_output(input_string: str) -> List[str]:
    """
    Parses a string into a list of strings.

    Parameters:
    input_string (str): The input string to parse.

    Returns:
    List[str]: The parsed output list.
    """

    input_list = input_string.split("\n")  # split the input by newline characters
    output_list = []
    for i, item in enumerate(input_list):
        item = item.lstrip(
            "0123456789. "
        )  # remove numbers and any leading whitespace
        if item:  # skip empty items
            output_list.append(item)
    return output_list

def get_device() -> torch.device:
    """
    Get the current device.

    If a CUDA GPU is available, it is used. Else, it checks for MPS compatibility. 
    If not, it defaults to the CPU.

    Returns:
    torch.device: The device to use for computations.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    return device

def read_json(path: Path) -> List[Dict[str, Union[str, int]]]:
    """
    Reads a JSON file and returns it as a Python list of dictionaries.

    Parameters:
    path (Path): The path to the JSON file to read.

    Returns:
    List[Dict[str, Union[str, int]]]: The content of the JSON file as a Python list of dictionaries.
    """
    with open(path, "r") as f:
        data: List[Dict[str, Union[str, int]]] = json.load(f)
    return data

def save_json(path: Path, container: Iterable) -> None:
    """
    Saves a Python iterable (such as a list or a dictionary) into a JSON file.

    Parameters:
    path (Path): The path to the JSON file to write into.
    container (Iterable): The Python iterable to write into the JSON file.
    """
    print(f"Saving json to {path}")
    with open(path, "w") as outfile:
        json.dump(container, outfile, ensure_ascii=False, indent=4)