from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys 

sys.path.append("../")

def load_model(model_path, tokenizer_name = "bert-base-uncased", device= "cuda:1"):
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer