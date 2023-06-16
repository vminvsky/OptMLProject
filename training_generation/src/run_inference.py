from prompting.prompts_langchain import DataTemplates
from config import AugmentConfig, DATA_DIR, SENTIMENT_DATA_DIR, SARCASM_DATA_DIR

from prompting.prompt_module import PromptModule

import json
import os
import time
from typing import Callable, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import openai

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import pandas as pd

from tqdm import tqdm
from utils import parse_output

import hydra
import logging

load_dotenv() # Load environment variables
logger = logging.getLogger(__name__)


def generate_data(idx: int, input_text: str, polarize_directions: List[str], dataset: pd.DataFrame,
                  cfg: AugmentConfig) -> pd.DataFrame:
    """
    Generates synthetic data for each polarization direction.

    Parameters:
    idx (int): The index of the original text in the dataset.
    input_text (str): The original text used to generate the synthetic data.
    polarize_directions (List[str]): A list of directions to polarize the text.
    dataset (pd.DataFrame): The original dataset.
    cfg (AugmentConfig): The configuration object for the data generation process.

    Returns:
    pd.DataFrame: A DataFrame containing the original and generated text, the polarization direction and the target label.
    """

    # Create an empty DataFrame with specified columns
    df = pd.DataFrame(columns=[f"text", "target", "labels", f"augmented_text"])
    # Loop over all polarization directions
    for direction in polarize_directions:
        # If in debug mode, stop generating after 40 iterations
        if cfg.experiment.debug:
            if idx >= 40:
                break
        
        # Try generating new data using the PromptModule
        try:
            generator = PromptModule(cfg, input_text, **{"direction": direction})
            final_output = generator.generate()
        # If the request times out, log an error message and continue to the next iteration
        except openai.error.Timeout as e:
            logger.error("OpenAI request timed out. Skipping this entry.")
            continue
        # Parse the output from the generator
        augmented_text = parse_output(input_string=final_output)

        # Create a new DataFrame with the generated data
        pl = pd.DataFrame(augmented_text, columns=[f"augmented_text"])
        pl["text"] = input_text
        pl["labels"] = direction
        pl["target"] = dataset[cfg.experiment.label_name][idx]
        # Concatenate the new DataFrame with the existing one
        df = pd.concat([df, pl], ignore_index=True, axis=0)
    return df


@hydra.main(version_base=None, config_path="conf", config_name="inference_root.yaml")
def main(cfg: AugmentConfig) -> None:
    """
    The main function of the script, responsible for managing the data generation process.

    The function loads the original dataset, initializes a thread pool, submits data generation tasks to the thread pool,
    waits for the tasks to complete, concatenates the results and saves the final dataset to a JSON file.

    Parameters:
    cfg (AugmentConfig): The configuration object for the data generation process.

    Returns:
    None
    """
    current_time_stamp = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    start_time = time.time()
    augmentation_templates = DataTemplates()

    task = cfg.experiment.task
    # Load the base dataset
    dataset = pd.read_json(os.path.join(cfg.file_dir, "base.json"))

    polarize_directions = cfg.experiment.directions

    num_keys = len(cfg.local.OPENAI_API_KEY)
    logger.info(f"Generating data using {cfg.experiment.num_threads} threads across {num_keys} keys.")

    logger.info(f"Dataset is {len(dataset)} rows")
    # Initialize a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_keys*cfg.experiment.num_threads) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(generate_data, idx, input_text, polarize_directions, dataset, cfg)
                   for idx, input_text in dataset["text"].items()]
    # Calculate and log the execution time
    rounded_time = round(time.time() - start_time)
    content = f"Execution time: {rounded_time} (s)"
    logger.info(content)

    # Calculate and log the execution time
    df = pd.concat([future.result() for future in futures], ignore_index=True, axis=0)

    df = df.reset_index(drop=True)

    df.to_json(
        os.path.join(cfg.output_dir, f"{cfg.experiment.prompt_label}_{cfg.experiment.models.model_name}.json"),
        orient="records",
        indent=2
    )

# Entry point for the script
if __name__ == "__main__":
    main()
