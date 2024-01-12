# BusinessBERT
Industry-sensitive Language Model for Business. The model is available on HuggingFace: https://huggingface.co/pborchert/BusinessBERT

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("pborchert/BusinessBERT")
```

## Summary
- Pretrained Transformer: BERT-Base architecture 
- Trained on business communication extracted:
  - Management Discussion and Analaysis statements [CaltechDATA | MD&A](https://data.caltech.edu/records/1249)
  - Company Website content [This study | CompanyWeb](https://huggingface.co/datasets/pborchert/CompanyWeb)
  - Scientific Business Literature [Semantic Scholar | S2ORC](https://api.semanticscholar.org/corpus)
- Additional pretraining objective: Industry classification (IC) predicting the standard industry classification textual documents originate from
- SOTA performance on business related text classification, named entity recognition and question answering benchmarks

## Abstract
We introduce BusinessBERT, a new industry-sensitive language model for business applications. The key novelty of our model lies in incorporating industry information to enhance decision-making in business-related natural language processing (NLP) tasks. BusinessBERT extends the Bidirectional Encoder Representations from Transformers (BERT) architecture by embedding industry information during pretraining through two innovative approaches that enable BusinessBert to capture industry-specific terminology: (1) BusinessBERT is trained on business communication corpora totaling 2.23 billion tokens consisting of company website content, MD&A statements and scientific papers in the business domain; (2) we employ industry classification as an additional pretraining objective. Our results suggest that BusinessBERT improves data-driven decision-making
by providing superior performance on business-related NLP tasks. Our experiments cover 7 benchmark datasets that include text classification, named entity recognition, sentiment analysis, and question-answering tasks. Additionally, this paper reduces the complexity of using BusinessBERT for other NLP applications by making it freely available as a pretrained language model to the business community.

## Benchmark
The benchmark consists of business related NLP tasks structured in the following categories:

*Text classification*
- Risk: Financial risk classification based corporate disclosures. [Link](https://pubsonline.informs.org/doi/10.1287/mnsc.2014.1930)
- News: Topic classification based on news headlines. [Link](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection)

*Named Entity Recognition*
- SEC filings: NER based on financial agreements. [Link](https://people.eng.unimelb.edu.au/tbaldwin/resources/finance-sec)

*Sentiment Analysis*
- FiQA: Predict continuous sentiment score based on microblog messages, news statements or headlines. Run `data/fiqa/build_fiqa.py` to combine the data parts and create `data/fiqa/train.json`. [Link](https://sites.google.com/view/fiqa/home) or [Direct Download](https://drive.google.com/file/d/1icRTdnu8UcWyDIXtzpsYc2Hm6ACHT-Ch/view)
- Financial Phrasebank: Sentiment classification based on financial news. [Link](https://huggingface.co/datasets/financial_phrasebank)
- StockTweets: Sentiment classification based on tweets using tags like "#SPX500" and "#stocks". [Link](https://ieee-dataport.org/open-access/stock-market-tweets-data)

*Question Answering*
- FinQA: Generative question answering based on earnings reports of S\&P 500 companies. [Link](https://github.com/czyssrs/finqa)



## Folder structure

Run `makfolder.sh` to create the folder structure below.

```sh
BusinessBERT
├───data
│   ├───finphrase # obsolete, load data directly from https://huggingface.co/datasets
│   ├───fiqa
│   │       task1_headline_ABSA_train.json
│   │       task1_post_ABSA_train.json
│   │       build_fiqa.py
│   │       train.json
│   │
│   ├───news # obsolete, load data directly from https://huggingface.co/datasets
│   ├───risk
│   │       groundTruth.dat
│   │
│   ├───secfilings
│   │       test.txt
│   │       train.txt
│   │       valid.txt
│   │
│   └───stocktweets
│           tweets_clean.csv
│
└───tasks
        finphrase.py
        fiqa.py
        news.py
        risk.py
        secfilings.py
        stocktweets.py
        __init__.py
```

## Code

The business NLP benchmark results can be replicated using the `run_benchmark.sh` script. Note that the FinQA dataset and corresponding code is available here: [https://github.com/czyssrs/finqa](https://github.com/czyssrs/finqa)

```sh
for task in "risk" "news" "secfilings" "fiqa" "finphrase" "stocktweets"
do
    for model in "pborchert/BusinessBERT" "bert-base-uncased" "ProsusAI/finbert" "yiyanghkust/finbert-pretrain"
    do
        for seed in 42
        do 
            python businessbench.py \
            --task_name $task \
            --model_name $model \
            --seed $seed
        done
    done
done
```
The batch size and gradient accumulation parameters are selected for running the experiment on a NVIDIA RTX4000 (8GB) GPU.


## License 
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.