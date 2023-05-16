from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import hydra 
import os 
from utils.processing_files import process_col_names
import pandas as pd 
    
def _load_data(cfg, dataset, tokenizer, name=""):
    MAX_LEN = 128
    train_data = hydra.utils.instantiate(cfg.datamodule, df = dataset, tokenizer=tokenizer, max_len=MAX_LEN)
    if name == "":
        train_loader = DataLoader(train_data, batch_size=cfg.model.batch_size, 
                            num_workers=1, shuffle=cfg.experiment.shuffle, pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=cfg.model.batch_size, 
                            num_workers=1, shuffle=False, pin_memory=True)
    return train_loader


def data_processing(cfg, training_points):
    debug = cfg.model.debug
    # Load train and test datasets
    
    if debug:
        train_df = pd.read_csv(os.path.join(cfg.data_processing.file_dir, cfg["training_name"])).dropna().reset_index(drop=True).head(100)
    else:
        if training_points == "all":
            train_df = pd.read_csv(os.path.join(cfg.data_processing.file_dir, cfg["training_name"])).dropna().reset_index(drop=True)
        elif type(training_points) == int:
            train_df = pd.read_csv(os.path.join(cfg.data_processing.file_dir, cfg["training_name"])).dropna().reset_index(drop=True).head(training_points)
        else:
            raise TypeError("Training points needs to be into or 'all'")

    validation = pd.read_csv(os.path.join(cfg.data_processing.file_dir, cfg["validation_name"]))

    if cfg.model.equalize_test:
        test_df = pd.read_csv(os.path.join(cfg.data_processing.file_dir, cfg.testing_name)).dropna().groupby("label_name").sample(150).reset_index(drop=True)
        validation = pd.read_csv(os.path.join(cfg.data_processing.file_dir, cfg.validation_name)).dropna().groupby("label_name").sample(150).reset_index(drop=True)        
    else:
        test_df = pd.read_csv(os.path.join(cfg.data_processing.file_dir, cfg.testing_name)).dropna().reset_index(drop=True)
    
    train_df, validation,test_df = process_col_names(train_df,validation, test_df, cfg)
    

    label2id = {label: i for i, label in enumerate(cfg.experiment.labels)}
    train_df['label'] = train_df['label'].apply(lambda x: label2id[x])
    test_df['label'] = test_df['label'].apply(lambda x: label2id[x])
    validation['label'] = validation['label'].apply(lambda x: label2id[x])
    return train_df, validation, test_df