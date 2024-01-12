import torch 
from torch.utils.data import Dataset
from transformers import BertForTokenClassification, RobertaForTokenClassification, BertTokenizer, RobertaTokenizer
import numpy as np
from seqeval.metrics import classification_report

def get_params(**kwargs):
    # default parameters
    params = {
        "lr":1e-4,
        "max_len":256,
        "batch_size":16,
        "grad_accum":1,
        "epochs":10,
        "do_lower_case":True,
        "path_train":"./data/secfilings/train.txt",
        "path_valid":"./data/secfilings/valid.txt",
        "path_test":"./data/secfilings/test.txt",
    }

    # user defined parameters
    for p in kwargs:
        if kwargs.get(p) is not None:
            params[p] = kwargs.get(p)
    return params

def get_labellist():
    id2label = {
        0:"O",
        1:"I-PER",
        2:"I-ORG",
        3:"I-LOC",
        4:"I-MISC",
    }

    label2id = {v:k for k,v in id2label.items()}
    return id2label, label2id

class DatasetSecFilings(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        super().__init__()

        self.X = X
        self.y = y
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.CLS = tokenizer.cls_token_id
        self.SEP = tokenizer.sep_token_id
        self.PAD = tokenizer.pad_token_id
        self.IGNORE_INDEX = 0

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):

        # get batch
        text = self.X[index]
        label = self.y[index]

        # process text
        enc_input_ids = torch.empty((self.max_length,), dtype=torch.long).fill_(self.PAD)
        enc_labels = torch.zeros((self.max_length,), dtype=torch.long)
        enc_attention_mask = torch.zeros((self.max_length,), dtype=torch.long)

        inputs_ids = [self.CLS]
        labels = [self.IGNORE_INDEX]
        # tokenize words corresponding to each label
        for i in range(len(text)):
            tokens = self.tokenizer(text[i], return_attention_mask=False, return_token_type_ids=False, add_special_tokens=False)["input_ids"]
            inputs_ids += tokens
            labels.append(label[i])
            # append zeros for words encoded in multiple token
            if len(tokens) > 1:
                for _ in range(1,len(tokens)):
                    labels.append(self.IGNORE_INDEX)
        inputs_ids.append(self.SEP)
        labels.append(self.IGNORE_INDEX)

        enc_input_ids[:len(inputs_ids)] = torch.as_tensor(inputs_ids[:self.max_length])
        enc_labels[:len(labels)] = torch.as_tensor(labels[:self.max_length])
        enc_attention_mask[:len(inputs_ids)] = torch.ones((len(inputs_ids[:self.max_length],)), dtype=torch.long)

        encoded = {"input_ids":enc_input_ids, "attention_mask":enc_attention_mask, "labels":enc_labels}
        return encoded

def load_file(path, label2id):
    with open(path, encoding="utf-8") as fp:
        lines = fp.readlines()

    lines = [i.split(" ") for i in lines]

    docs = [[]]
    labels = [[]]
    for line in lines:
        if (line[0] == "-DOCSTART-") or (line[0] == "\n"):
            if len(docs[-1]) > 0:
                docs.append([])
                labels.append([])
        elif line[0]:
            docs[-1].append(line[0])
            labels[-1].append(label2id[line[-1].replace("\n", "")])
        else:
            pass
    
    docs.pop()
    labels.pop()
    
    return docs, labels

def get_dataset(train, valid, test, tokenizer, max_len):
    data_train  = DatasetSecFilings(X=train[0], y=train[1], tokenizer=tokenizer, max_length=max_len)
    data_valid  = DatasetSecFilings(X=valid[0], y=valid[1], tokenizer=tokenizer, max_length=max_len)
    data_test  = DatasetSecFilings(X=test[0], y=test[1], tokenizer=tokenizer, max_length=max_len)

    return data_train, data_valid, data_test

def get_model(model_name, id2label, label2id):
    if model_name in ["bert-base-uncased", "ProsusAI/finbert", "yiyanghkust/finbert-pretrain", "pborchert/BusinessBERT"]:
        model = BertForTokenClassification.from_pretrained(model_name, use_auth_token=True, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    elif model_name.startswith("roberta"):
        model = RobertaForTokenClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id)
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

def make_compute_metrics(id2label):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        pred = np.argmax(logits, axis=-1)
        map_func = lambda x: [id2label[i] for i in x]

        labels = list(map(map_func, labels))
        pred = list(map(map_func, pred))

        res_dict = classification_report(labels, pred, output_dict=True, digits=4)

        return res_dict
    return compute_metrics

def eval2table(x):
    rows = []
    for k,v in x.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                rows.append([f"{k}_{kk}", vv])
        else:
            rows.append([k, v])
    return rows

def load(**kwargs):
    torch.manual_seed(kwargs.get("seed"))
    id2label, label2id = get_labellist()
    train = load_file(path=kwargs.get("path_train"), label2id=label2id)
    valid = load_file(path=kwargs.get("path_valid"), label2id=label2id)
    test = load_file(path=kwargs.get("path_test"), label2id=label2id)
    tokenizer = get_tokenizer(model_name=kwargs.get("model_name"), do_lower_case=kwargs.get("do_lower_case"))
    model = get_model(model_name=kwargs.get("model_name"), id2label=id2label, label2id=label2id)
    data_train, data_valid, data_test = get_dataset(train=train, valid=valid, test=test, tokenizer=tokenizer, max_len=kwargs.get("max_len"))
    return (data_train, data_valid, data_test), model, tokenizer, make_compute_metrics(id2label=id2label)
