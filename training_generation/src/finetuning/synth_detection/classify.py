import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

device = "cuda:0"
batch_size = 128


def classify(model, tokenizer, texts, device="cuda:0"):
    # Set the model to evaluate mode
    model.eval()

    # Tokenize the input sentences and create input tensors for the model
    inputs = tokenizer(
        texts,
        truncation=True,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_tensors="pt"
    )

    # Pass the inputs through the model to get the predicted labels
    labels = []
    with torch.no_grad():
        outputs = model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
        labels.append(outputs.logits.detach().cpu().numpy())

    return labels


def datapoints(df, model, tokenizer):
    labels = []
    batch_texts = []
    for i, row in df.iterrows():
        batch_texts.append(row["augmented_text"])
        if len(batch_texts) == batch_size:
            labels.extend(classify(model, tokenizer, batch_texts))
            batch_texts = []
    if batch_texts:  # process the remaining items
        labels.extend(classify(model, tokenizer, batch_texts))
    return labels


if __name__ == "__main__":
    filter = False
    # df = pd.read_csv("/scratch/venia/socialgpt/SocialSynth/data/sarcasm/synth/Train-SynthReal.csv").sample(10)
    # df = pd.read_json("/scratch/venia/socialgpt/venia_worker_vs_gpt/data/sarcasm/simple_gpt-3.5-turbo.json", orient="records").sample(1000)
    # df = pd.read_json("/scratch/venia/socialgpt/venia_worker_vs_gpt/data/sarcasm/grounded_norewrite_gpt-3.5-turbo.json", orient="records").sample(1000)
    df = pd.read_json("/scratch/venia/socialgpt/venia_worker_vs_gpt/data/sarcasm/grounded_gpt-3.5-turbo.json", orient="records")
    df = pd.read_json("/scratch/venia/socialgpt/venia_worker_vs_gpt/data/sarcasm/filtered_gpt-3.5-turbo.json", orient="records")

    # df = pd.read_json("/scratch/venia/socialgpt/venia_worker_vs_gpt/data/sarcasm/taxonomy_mandatory_gpt-3.5-turbo.json", orient="records").sample(1000)
    #df = pd.read_json("/scratch/venia/socialgpt/venia_worker_vs_gpt/data/sarcasm/base.json", orient="records").head(500)
    model = AutoModelForSequenceClassification.from_pretrained(
        "/scratch/venia/socialgpt/SocialSynth/logs/inference/runs/sarcasm_true_vs_synth/2023-04-14_12-11-25/model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    labels = datapoints(df, model, tokenizer)
    labels = np.concatenate(labels, axis=0)  # concatenate all batched results
    df["synth_logit"] = labels[:, 1]
    print("Accuracy: ", df[df["synth_logit"] < 0].count() / df.count())
    print(df.head(5))
    if filter:
        filtered = df[df["synth_logit"] < 1.5]
        print(len(filtered))
        filtered[["text", "target", "labels", "augmented_text"]].to_json("/scratch/venia/socialgpt/venia_worker_vs_gpt/data/sarcasm/filtered_gpt-3.5-turbo.json", orient="records")

