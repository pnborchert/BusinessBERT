import torch 
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import warnings
import datasets
from transformers import Trainer

def get_params(**kwargs):
    # default parameters
    params = {
        "lr":1e-5,
        "max_len":512,
        "batch_size":4,
        "grad_accum":4,
        "epochs":20,
        "do_lower_case":True,
        "path":("reuters21578", "ModHayes"),
        "target":"topics",
    }

    # user defined parameters
    for p in kwargs:
        if kwargs.get(p) is not None:
            params[p] = kwargs.get(p)

    return params

class DatasetNews(Dataset):
    def __init__(self, X, y, tokenizer, max_length, num_labels):
        super().__init__()

        self.X = X
        self.y = y
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.num_labels = num_labels

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

def load_file(seed, path, target):
    dataset = datasets.load_dataset(path[0], path[1])
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    # filter rows without text 
    mask = list(df_train["text"].map(lambda x: len(x)) > 0)
    df_train = df_train.loc[mask,:].reset_index(drop=True)

    mask = list(df_test["text"].map(lambda x: len(x)) > 0)
    df_test = df_test.loc[mask,:].reset_index(drop=True)

    # filter occurence >= 20
    val, occ = np.unique(np.hstack(df_train[target]), return_counts=True)
    set_target = val[occ >= 20]

    lookup = dict(zip(set_target, np.arange(0,len(set_target)+1,1)))

    # transform train and test array
    arr_train = df_train[target].map(lambda lst:[lookup[item] for item in lst if item in lookup.keys()] if len(lst) > 0 else np.nan)
    arr_test = df_test[target].map(lambda lst:[lookup[item] for item in lst if item in lookup.keys()] if len(lst) > 0 else np.nan)

    # one-hot matrix train
    num_labels = len(lookup.keys())
    dim_row_train = df_train.shape[0]
    matrix_train = np.zeros((dim_row_train,num_labels))

    for i, val in enumerate(arr_train): 
        if isinstance(val, list):
            np.put(matrix_train[i], val, 1)


    # one-hot matrix test
    dim_row_test = df_test.shape[0]
    matrix_test = np.zeros((dim_row_test,num_labels))

    for i, val in enumerate(arr_test): 
        if isinstance(val, list):
            np.put(matrix_test[i], val, 1)

    # mimick 10-fold cv split 
    X, y = np.asarray(df_train["text"].map(lambda x: x.replace("\n", " "))), matrix_train
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, random_state=seed)

    return (X_train, y_train), (X_valid, y_valid), (np.asarray(df_test["text"].map(lambda x: x.replace("\n", " "))), matrix_test), num_labels

def get_dataset(train, valid, test, tokenizer, max_len, num_labels):
    data_train  = DatasetNews(X=train[0], y=train[1], tokenizer=tokenizer, max_length=max_len, num_labels=num_labels)
    data_valid  = DatasetNews(X=valid[0], y=valid[1], tokenizer=tokenizer, max_length=max_len, num_labels=num_labels)
    data_test  = DatasetNews(X=test[0], y=test[1], tokenizer=tokenizer, max_length=max_len, num_labels=num_labels)

    return data_train, data_valid, data_test

def get_model(model_name, num_labels):
    if model_name in ["bert-base-uncased", "ProsusAI/finbert", "yiyanghkust/finbert-pretrain", "pborchert/BusinessBERT"]:
        model = BertForSequenceClassification.from_pretrained(model_name, use_auth_token=True, num_labels=num_labels, ignore_mismatched_sizes=True, problem_type="multi_label_classification")
    elif model_name.startswith("roberta"):
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")
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
    logits, labels = eval_preds
    pred = torch.as_tensor(logits).sigmoid().numpy().astype(int)

    # print(np.bincount(pred))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy             = accuracy_score(labels, pred)
        f1_weighted          = f1_score(labels, pred, average="weighted")
        precision_weighted   = precision_score(labels, pred, average="weighted")
        recall_weighted      = recall_score(labels, pred, average="weighted")
        f1_micro             = f1_score(labels, pred, average="micro")
        precision_micro      = precision_score(labels, pred, average="micro")
        recall_micro         = recall_score(labels, pred, average="micro")

    res_dict = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        # 'precision_weighted': precision_weighted,
        # 'recall_weighted': recall_weighted,
        # 'f1_micro': f1_micro,
        # 'precision_micro': precision_micro,
        # 'recall_micro': recall_micro,
    }

    return res_dict

def load(**kwargs):
    train, valid, test, num_labels = load_file(seed=kwargs.get("seed"), path=kwargs.get("path"), target=kwargs.get("target"))
    tokenizer = get_tokenizer(model_name=kwargs.get("model_name"), do_lower_case=kwargs.get("do_lower_case"))
    model = get_model(model_name=kwargs.get("model_name"), num_labels=num_labels)
    data_train, data_valid, data_test = get_dataset(train=train, valid=valid, test=test, tokenizer=tokenizer, max_len=kwargs.get("max_len"), num_labels=num_labels)
    return (data_train, data_valid, data_test), model, tokenizer, compute_metrics