import torch 
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import warnings
from transformers import Trainer

def get_params(**kwargs):
    # default parameters
    params = {
        "lr":5e-5,
        "max_len":64,
        "batch_size":16,
        "grad_accum":1,
        "epochs":20,
        "do_lower_case":True,
        "path":"./data/fiqa/train.json",
    }

    # user defined parameters
    for p in kwargs:
        if kwargs.get(p) is not None:
            params[p] = kwargs.get(p)

    return params

class DatasetFiQA(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        super().__init__()

        self.X = X
        self.y = y
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):

        # get batch
        text = self.X[index]
        label = self.y[index]

        # process text
        encoded = self.tokenizer.encode_plus(text, return_attention_mask=True, add_special_tokens=False, return_token_type_ids=False, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        for key,val in encoded.items(): encoded[key] = val.squeeze(0)
        encoded["labels"] = torch.as_tensor(label, dtype=torch.float)
        
        return encoded

def load_file(seed, path):
    df = pd.read_json(path)

    # mimick 10-fold cv split 
    X, y = np.asarray(df["sentence"]), np.asarray(df["sentiment_score"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, random_state=seed)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def get_dataset(train, valid, test, tokenizer, max_len):
    data_train  = DatasetFiQA(X=train[0], y=train[1], tokenizer=tokenizer, max_length=max_len)
    data_valid  = DatasetFiQA(X=valid[0], y=valid[1], tokenizer=tokenizer, max_length=max_len)
    data_test  = DatasetFiQA(X=test[0], y=test[1], tokenizer=tokenizer, max_length=max_len)

    return data_train, data_valid, data_test

def get_model(model_name):
    if model_name in ["bert-base-uncased", "ProsusAI/finbert", "yiyanghkust/finbert-pretrain", "pborchert/BusinessBERT"]:
        model = BertForSequenceClassification.from_pretrained(model_name, use_auth_token=True, problem_type="regression", num_labels=1, ignore_mismatched_sizes=True)
    elif model_name.startswith("roberta"):
        model = RobertaForSequenceClassification.from_pretrained(model_name, problem_type="regression", num_labels=1)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not supported.")
    return model

def get_tokenizer(model_name, do_lower_case):
    if model_name in ["bert-base-uncased", "ProsusAI/finbert", "yiyanghkust/finbert-pretrain", "pborchert/BusinessBERT"]:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case, use_auth_token=True)
    elif model_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not supported.")
    return tokenizer

def compute_metrics(eval_preds):
    pred, labels = eval_preds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mape = mean_absolute_percentage_error(labels, pred)
        mse  = mean_squared_error(labels, pred)
        mae  = mean_absolute_error(labels, pred)

    res_dict = {
        'mape': mape,
        'mse': mse,
        'mae': mae
    }

    return res_dict

def load(**kwargs):
    train, valid, test =  load_file(seed=kwargs.get("seed"), path=kwargs.get("path"))
    tokenizer = get_tokenizer(model_name=kwargs.get("model_name"), do_lower_case=kwargs.get("do_lower_case"))
    model = get_model(model_name=kwargs.get("model_name"))
    data_train, data_valid, data_test = get_dataset(train=train, valid=valid, test=test, tokenizer=tokenizer, max_len=kwargs.get("max_len"))
    return (data_train, data_valid, data_test), model, tokenizer, compute_metrics