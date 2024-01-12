import pandas as pd

headline = pd.read_json("task1_headline_ABSA_train.json").T
post = pd.read_json("task1_post_ABSA_train.json").T

snippets = []
target = []
sentiment_score = []
aspects = []

for item in headline["info"]:
    snip, targ, sent, asp = item[0].values()
    snippets.append(snip)
    target.append(targ)
    sentiment_score.append(sent)
    # aspects.append(eval(asp)[0].split("/"))

headline["snippets"] = snippets
headline["target"] = target
headline["sentiment_score"] = sentiment_score
# headline["aspects"] = aspects

snippets = []
target = []
sentiment_score = []
aspects = []

for item in post["info"]:
    snip, sent, targ, asp = item[0].values()
    snippets.append(snip)
    target.append(targ)
    sentiment_score.append(sent)
    # if asp[-2] == "'":
    #     aspects.append(eval(asp)[0].split("/"))
    # else:
    #     aspects.append(eval(asp[:-1]+"']")[0].split("/"))

post["snippets"] = snippets
post["target"] = target
post["sentiment_score"] = sentiment_score
# post["aspects"] = aspects

fiqa = headline.append(post)
fiqa["sentiment_score"] = fiqa["sentiment_score"].astype(float)
fiqa = fiqa.sample(frac=1).reset_index(drop=True)

fiqa.to_json("train.json", orient="records")