import argparse, os, wandb
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import tasks

# parse input arguments
parser=argparse.ArgumentParser()
# required
parser.add_argument('--task_name', help='name of supported task', type=str, required=True)
parser.add_argument('--model_name', help='path to model on https://huggingface.co/models', type=str, required=True)
parser.add_argument('--seed', help='seed', type=int, default=42)
parser.add_argument('--output_dir', help='output directory for trained checkpoints', type=str, default="./runs")
parser.add_argument('--wandb_dir', help='local wandb directory', type=str, default="./wandb")
parser.add_argument('--wandb_project', help='wandb project prefix', type=str, default="businessbench")
# optional -> overwrite defaults
parser.add_argument('--do_early_stopping', dest='do_early_stopping', help='do_early_stopping', action="store_true", default=None)
parser.add_argument('--no_early_stopping', dest='do_early_stopping', help='do_early_stopping', action="store_false", default=None)
parser.add_argument('--early_stopping_patience', help='early_stopping_patience', type=int, default=2)

parser.add_argument('--lr', help='learning rate', type=float)
parser.add_argument('--max_len', help='max sequence length', type=int)
parser.add_argument('--batch_size', help='training batch size', type=int)
parser.add_argument('--grad_accum', help='gradient accumulation steps', type=int)
parser.add_argument('--epochs', help='training epochs', type=int)
parser.add_argument('--do_lower_case', help='lower case tokenizer', type=bool)
parser.add_argument('--path', help='filepath to dataset', type=str)
parser.add_argument('--path_train', help='filepath to train set', type=str)
parser.add_argument('--path_valid', help='filepath to validation set', type=str)
parser.add_argument('--path_test', help='filepath to test set', type=str)
parser.add_argument('--target', help='column containing target values (only applicable for news dataset)', type=str)

parser.add_argument('--weight_decay', help='train args weight_decay', type=float, default=0.0)
parser.add_argument('--do_train', dest='do_train', help='train args do_train', action="store_false")
parser.add_argument('--do_eval', dest='do_eval', help='train args do_eval', action="store_false")
parser.add_argument('--do_predict', dest='do_predict', help='train args do_predict', action="store_false")
parser.add_argument('--evaluation_strategy', help='evaluation_strategy', type=str, default="epoch")
parser.add_argument('--lr_scheduler_type', help='lr_scheduler_type', type=str, default="linear")
parser.add_argument('--save_strategy', help='save_strategy', type=str, default="epoch")
parser.add_argument('--save_steps', help='save_steps', type=int, default=500)
parser.add_argument('--report_to', help='report_to', type=str, default="wandb")
parser.add_argument('--save_total_limit', help='save_total_limit', type=int, default=1)
parser.add_argument('--max_steps', help='max_steps', type=int, default=-1)
parser.add_argument('--warmup_ratio', help='warmup_ratio', type=float, default=0.0)
parser.add_argument('--warmup_steps', help='warmup_steps', type=int, default=0)
parser.add_argument('--load_best_model_at_end', dest='load_best_model_at_end', help='load_best_model_at_end', action="store_true")
parser.add_argument('--metric_for_best_model', help='metric_for_best_model', type=str, default="loss")
args=vars(parser.parse_args())

def main(args):
    if args["task_name"] in tasks.supported_tasks:
        task = tasks.supported_tasks[args["task_name"]]
    else:
        raise NotImplementedError(f"Task '{args['task_name']}' is not supported.")

    args = task.get_params(**args)
    # fix parameter dependencies
    if ("do_early_stopping" in args) and (args["do_early_stopping"]):
        args["load_best_model_at_end"] = True

    os.makedirs(args["output_dir"], exist_ok=True)

    # load and process data
    data, model, tokenizer, compute_metrics = task.load(**args)
    dataset_train, dataset_valid, dataset_test = data

    # setup wandb
    wandb.init(
        project=f'{args["wandb_project"]}-{args["task_name"]}',
        name=f'{args["model_name"].split("/")[-1]}_{args["seed"]}',
        dir=args["wandb_dir"],
        reinit=True,
    )

    # setup trainer
    training_args = TrainingArguments(
        seed=args["seed"],
        output_dir=os.path.join(args["output_dir"], args["task_name"], f'{args["model_name"].split("/")[-1]}_{args["seed"]}'),
        do_train=args["do_train"],
        do_eval=args["do_eval"],
        do_predict=args["do_predict"],
        evaluation_strategy=args["evaluation_strategy"], 
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
        gradient_accumulation_steps=args["grad_accum"],
        learning_rate=args["lr"],
        weight_decay=args["weight_decay"],
        warmup_ratio=args["warmup_ratio"],
        warmup_steps=args["warmup_steps"],
        num_train_epochs=args["epochs"],
        max_steps=args["max_steps"],
        lr_scheduler_type=args["lr_scheduler_type"],
        save_strategy=args["save_strategy"],
        save_steps=args["save_steps"],
        save_total_limit=args["save_total_limit"],
        report_to=args["report_to"],
        run_name=f'{args["model_name"].split("/")[-1]}_{args["seed"]}',
        load_best_model_at_end=args["load_best_model_at_end"],
        metric_for_best_model=args["metric_for_best_model"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        compute_metrics=compute_metrics,
    )

    if ("do_early_stopping" in args) and (args["do_early_stopping"]):
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=args["do_early_stopping"])
        )

    # train
    trainer.train()

    # evaluate
    eval_test = trainer.evaluate(dataset_test)
    if args["task_name"] in ["secfilings"]:
        wandb.log({"test":wandb.Table(data=task.eval2table(eval_test),columns = ["metric", "value"])})
    else:
        wandb.log({"test":wandb.Table(data=list(eval_test.items()),columns = ["metric", "value"])})
    print(eval_test)

    wandb.finish()

if __name__ == "__main__":
    main(args)