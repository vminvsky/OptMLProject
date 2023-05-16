import pandas as pd 
import numpy as np
import jsonlines 
from omegaconf import OmegaConf, DictConfig
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import os
from datasets import Dataset
import hydra
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, AdamW, AutoModelForSequenceClassification
from sklearn import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle

from utils.data_loader import _load_data, data_processing
from utils.plot import plot

from torch.utils.data.dataloader import default_collate

import torch
import evaluate
import logging

log = logging.getLogger(__name__)

CUDA_NUM=1
device = torch.device(f'cuda:{CUDA_NUM}')


def train(cfg):
    # Initialize tokenizer
    training_points = cfg.experiment.training_points
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    log.info(f"Loading data")
    train_df, validation, test_df = data_processing(cfg, training_points)

    
    # Convert datasets to Hugging Face Datasets
    train_loader = _load_data(cfg, train_df,tokenizer)    
    validation_loader = _load_data(cfg, validation,tokenizer, "evaluate")    
    test_loader = _load_data(cfg, test_df,tokenizer, "evaluate")    

    # Initialize model and optimizer
    if "bert-base" in cfg.model.name:
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name, num_labels=len(cfg.experiment.labels))
        model.dropout.p = cfg.model.dropout
        for param in model.bert.parameters():
            param.requires_grad = cfg.model.unfreeze
    else:
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name, num_labels=len(cfg.experiment.labels))

    model.to(device)


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Num params in model: {pytorch_total_params}")
    
    accumulation_steps = cfg.model.training_arguments.gradient_accumulation_steps
    
    optimizer = AdamW(model.parameters(), lr=cfg.model.training_arguments.learning_rate)
    train_losses = []
    total_accuracies = []
    validation_losses = []
    total_f1s = []
    f1s = []
    probabilities = []
    # Train model
    for epoch in range(cfg.model.num_epochs):
        # Train on batches
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.int64)
            targets_onehot = F.one_hot(targets.to(torch.int64), len(cfg.experiment.labels)).float()


            outputs = model(ids, attention_mask=mask, labels=targets_onehot)
            probs = F.softmax(outputs.logits)
            probabilities.append([probs.cpu(), data["targets"].to("cpu")])
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_losses.append([epoch, loss.item()])

        log.info("Done training. Now evaluating.")
        model.eval()
        test_loss = 0
        predictions = []
        true_labels = []
        probabilities = []

        with torch.no_grad():
            for data in validation_loader:
                input_ids = data['ids'].to(device, dtype = torch.long)
                attention_mask = data['mask'].to(device, dtype = torch.long)
                labels = torch.tensor(data['targets']).to(device)
                targets_onehot = F.one_hot(labels.to(torch.int64), len(cfg.experiment.labels)).float()
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=targets_onehot)
                loss = outputs.loss
                test_loss += loss.item()
                logits = outputs.logits.detach().cpu().numpy()
                if cfg.experiment.callibration==True:
                    logits = np.exp(outputs.logits.detach().cpu().numpy())
                    if epoch>0:
                        logits = calibrator.predict_proba(logits)
                preds = np.argmax(logits, axis=1).tolist()
                
                predictions+=preds
                true_labels += labels.detach().cpu().numpy().tolist()
                # predictions += preds

                probabilities.append(logits)
                validation_losses.append([epoch, loss.item()])

        # Calculate evaluation metrics
        accuracy = metrics.accuracy_score(true_labels, predictions)
        total_accuracies.append(accuracy)
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        print(confusion_matrix)
        if len(cfg.experiment.labels) <= 2:
            precision, recall, f1_binary, _ = metrics.precision_recall_fscore_support(true_labels, predictions, average='binary', pos_label=1)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(true_labels, predictions, average='macro')
        total_f1s.append([epoch,f1])
        f1s.append(f1)
        
        if f1 >= np.max(f1s):
            log.info("Best model: Saving")
            
            if cfg.experiment.callibration==True:
                log.info("Training callibration")
                calibrator = LogisticRegression()
                all_probabilities = np.concatenate(probabilities, axis=0)
                calibrator.fit(all_probabilities, true_labels)
        
            model.save_pretrained(cfg.model_save_path)
            pred_final = predictions
            
        # Log results
        print(f'Epoch {epoch+1}: Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test accuracy: {accuracy:.4f} | Test precision: {precision:.4f} | f1: {f1:.4f}')
    
    
    save_file(os.path.join(cfg.metric_path, "val-predictions.pkl"), predictions)
    save_file(os.path.join(cfg.metric_path, "val-train_loss.pkl"), train_losses)
    save_file(os.path.join(cfg.metric_path, "val-losses.pkl"), validation_losses)
    save_file(os.path.join(cfg.metric_path, "val-total_f1s.pkl"), total_f1s)
    save_file(os.path.join(cfg.metric_path, "val-probabilities.pkl"), probabilities)


    validation["pred"] = pred_final
    model =  BertForSequenceClassification.from_pretrained(cfg.model_save_path).to(device)    # load the best model
    
    model.eval()                                            
    test_loss = 0   
    predictions = []
    true_labels = []
    test_losses = []
    probabilities = []

    with torch.no_grad():
        for data in test_loader:
            input_ids = data['ids'].to(device, dtype = torch.long)
            attention_mask = data['mask'].to(device, dtype = torch.long)
            labels = torch.tensor(data['targets']).to(device)
            targets_onehot = F.one_hot(labels.to(torch.int64), len(cfg.experiment.labels)).float()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=targets_onehot)
            loss = outputs.loss
            test_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()                
            if cfg.experiment.callibration==True:
                probabilities = np.exp(logits).tolist()  # Convert logits to probabilities
                logits = calibrator.predict_proba(probabilities)
            preds = np.argmax(logits, axis=1).tolist()
            true_labels +=  labels.detach().cpu().numpy().tolist()
            predictions += preds
            test_losses.append(loss.item())
            
        # Calculate evaluation metrics
        accuracy = metrics.accuracy_score(true_labels, predictions)
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        print(confusion_matrix)
        if len(cfg.experiment.labels) <= 2:
            precision, recall, f1_binary, _ = metrics.precision_recall_fscore_support(true_labels, predictions, average='binary', pos_label=1)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(true_labels, predictions, average='macro')
        
        save_file(os.path.join(cfg.metric_path, "test-predictions.pkl"), predictions)
        save_file(os.path.join(cfg.metric_path, "test-losses.pkl"), test_losses)
        save_file(os.path.join(cfg.metric_path, "test-f1.pkl"), f1)
        if len(cfg.experiment.labels) <= 2:
            save_file(os.path.join(cfg.metric_path, "test-f1-binary.pkl"), f1_binary)
        save_file(os.path.join(cfg.metric_path, "test-accuracy.pkl"), accuracy)
            
        # Log results
        print(f'Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test accuracy: {accuracy:.4f} | Test precision: {precision:.4f} | f1: {f1:.4f}')
    

    with open(os.path.join(cfg.metric_path, "test_losses.pkl"), "wb") as f:
        pickle.dump(test_losses, f)
    validation.to_csv(os.path.join(cfg.metric_path,"validation_results.csv"), index=False)
    plot(total_accuracies, cfg.training_name, cfg.testing_name,os.path.join(cfg.metric_path, "accuracies.jpg"))
    plot(f1s, cfg.training_name, cfg.testing_name,os.path.join(cfg.metric_path, "f1s.jpg"))

def save_file(path,file):
    with open(path, "wb") as f:
        pickle.dump(file, f)

@hydra.main(version_base="1.2", config_path="../config", config_name="config_root")
def bert(cfg: DictConfig): 
    train(cfg)

if __name__=="__main__":
    torch.cuda.set_device(f"cuda:{CUDA_NUM}")
    torch.cuda.empty_cache()
    bert()