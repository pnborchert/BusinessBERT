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