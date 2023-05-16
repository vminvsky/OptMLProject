import json
import os
import time
from typing import Callable, Dict, List, Tuple


from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import pandas as pd

from tqdm import tqdm
from utils import parse_output

import hydra

from prompting.langchain import DataTemplates
from config import AugmentConfig, DATA_DIR, SENTIMENT_DATA_DIR



load_dotenv()

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="data_augmentation.yaml"
)
def main(cfg: AugmentConfig) -> None:
    augmentation_templates = DataTemplates()
    
    if cfg.dataset == "sentiment":
        dataset = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "base.json"))
        text = "text"
        description_prompt = augmentation_templates.sentiment_description_prompt()
        rewrite_prompt = augmentation_templates.sentiment_rewrite_prompt()
    else:
        raise ValueError(f"Only sentiment implemented right now.")
    
    temperature = .7
    
    df = pd.DataFrame(columns=[f"{text}", "target","labels", f"augmented_{text}", "original"])
    
    

    for idx, input_text in tqdm(dataset[text].items()):
        
        if cfg.debug_k==True:
            if idx >= 4:
                break
        llm = ChatOpenAI(model_name=cfg.model, temperature=temperature)
        
        if cfg.dataset=="sentiment":
            llm_chain = LLMChain(prompt=description_prompt, llm =llm)
            output = llm_chain.run({})
            
            final_response_chain = LLMChain(prompt=rewrite_prompt, llm=llm)
            
            final_output = final_response_chain.run({"characteristics": output, "text": input_text})
        else:
            raise NotImplementedError
        
        labels, augmented_text = parse_output(input_string=final_output)
        
        pl = pd.DataFrame(augmented_text, columns=[f"augmented_{text}"])
        pl[text] = input_text
        pl["labels"] = labels
        pl["target"] = dataset["target"][idx]
        pl["original"] = final_output
        
        df = pd.concat(
            [df, pl],
            ignore_index=True,
            axis=0
        )
        
    df = df.reset_index(drop=True)
    
    if cfg.dataset =="sentiment":
        df.to_json(
            SENTIMENT_DATA_DIR / f"{cfg.model}_augmented.json",
            orient="records"
        )
        
if __name__=="__main__":
    main()
        