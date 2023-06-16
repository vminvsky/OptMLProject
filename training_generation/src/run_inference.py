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

load_dotenv()
logger = logging.getLogger(__name__)


def generate_data(idx: int, input_text: str, polarize_directions: List[str], dataset: pd.DataFrame,
                  cfg: AugmentConfig) -> pd.DataFrame:
    df = pd.DataFrame(columns=[f"text", "target", "labels", f"augmented_text"])

    for direction in polarize_directions:
        if cfg.experiment.debug:
            if idx >= 40:
                break

        try:
            generator = PromptModule(cfg, input_text, **{"direction": direction})
            final_output = generator.generate()
        except openai.error.Timeout as e:
            logger.error("OpenAI request timed out. Skipping this entry.")
            continue

        augmented_text = parse_output(input_string=final_output)

        pl = pd.DataFrame(augmented_text, columns=[f"augmented_text"])
        pl["text"] = input_text
        pl["labels"] = direction
        pl["target"] = dataset[cfg.experiment.label_name][idx]

        df = pd.concat([df, pl], ignore_index=True, axis=0)
    return df


@hydra.main(version_base=None, config_path="conf", config_name="inference_root.yaml")
def main(cfg: AugmentConfig) -> None:
    current_time_stamp = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    start_time = time.time()
    augmentation_templates = DataTemplates()

    task = cfg.experiment.task

    dataset = pd.read_json(os.path.join(cfg.file_dir, "base.json"))

    polarize_directions = cfg.experiment.directions

    num_keys = len(cfg.local.OPENAI_API_KEY)
    logger.info(f"Generating data using {cfg.experiment.num_threads} threads across {num_keys} keys.")

    logger.info(f"Dataset is {len(dataset)} rows")
    with ThreadPoolExecutor(max_workers=num_keys*cfg.experiment.num_threads) as executor:
        futures = [executor.submit(generate_data, idx, input_text, polarize_directions, dataset, cfg)
                   for idx, input_text in dataset["text"].items()]

    rounded_time = round(time.time() - start_time)
    content = f"Execution time: {rounded_time} (s)"
    logger.info(content)

    df = pd.concat([future.result() for future in futures], ignore_index=True, axis=0)

    df = df.reset_index(drop=True)

    df.to_json(
        os.path.join(cfg.output_dir, f"{cfg.experiment.prompt_label}_{cfg.experiment.models.model_name}.json"),
        orient="records",
        indent=2
    )


if __name__ == "__main__":
    main()
