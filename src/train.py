import os
import sys
import json
import time
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader

from loss import MRCNERLoss
from trainer import Trainer
from model import MRCNERModel
from metrics import MRCNERMetrics
from dataset import MRCNERDataset
from optimization import configure_optimizers


class GeneralConfig():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_args():
    args_parser = ArgumentParser()

    args_parser.add_argument("--train_data_path", type=str, default="../data/MRC/conll03/mrc-ner.train")
    args_parser.add_argument("--eval_data_path", type=str, default="../data/MRC/conll03/mrc-ner.dev")
    args_parser.add_argument("--entity_types_path", type=str, default="../data/MRC/conll03/entity_types.json")
    args_parser.add_argument("--pretrained_model_path", type=str, default="../../pretrained_model/bert-base-uncased")
    args_parser.add_argument("--max_length", type=int, default=512)
    args_parser.add_argument("--train_batch_size", type=int, default=8)
    args_parser.add_argument("--eval_batch_size", type=int, default=8)
    args_parser.add_argument("--intermediate_size", type=int, default=768)
    args_parser.add_argument("--dropout", type=float, default=0.1)
    args_parser.add_argument("--device", type=str, default="cuda")
    args_parser.add_argument("--start_weight", type=float, default=1.0)
    args_parser.add_argument("--end_weight", type=float, default=1.0)
    args_parser.add_argument("--span_weight", type=float, default=1.0)
    args_parser.add_argument("--span_loss_candidate", type=str, default="pred_and_gold")
    args_parser.add_argument("--optimizer", type=str, default="torch.adam")
    args_parser.add_argument("--weight_decay", type=float, default=0.01)
    args_parser.add_argument("--lr", type=float, default=3e-5)
    args_parser.add_argument("--lr_mini", type=float, default=3e-7)
    args_parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    args_parser.add_argument("--num_train_epochs", type=int, default=20)
    args_parser.add_argument("--lr_scheduler", type=str, default="linear")
    args_parser.add_argument("--warmup_steps", type=int , default=0)
    args_parser.add_argument("--output_dir", type=str, default="../outputs/")

    args = args_parser.parse_args()
    
    return args


def train(args):
    train_data = json.load(open(args.train_data_path, encoding="utf-8"))
    eval_data = json.load(open(args.eval_data_path, encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    train_dataset = MRCNERDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    eval_dataset = MRCNERDataset(
        data=eval_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
    )

    pretrained_model_config = AutoConfig.from_pretrained(args.pretrained_model_path)
    hidden_size = pretrained_model_config.hidden_size
    model_config = GeneralConfig(
        pretrained_model_path=args.pretrained_model_path,
        hidden_size=hidden_size,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
    )

    device = torch.device(args.device)
    model = MRCNERModel(model_config)
    model = model.to(device)

    loss_computer = MRCNERLoss(
        start_weight=args.start_weight,
        end_weight=args.end_weight,
        span_weight=args.span_weight,
        span_loss_candidate=args.span_loss_candidate,
    )

    entity_types = json.load(open(args.entity_types_path))
    metrics_computer = MRCNERMetrics(
        entity_types=entity_types
    )

    optimizer, scheduler = configure_optimizers(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        lr=args.lr,
        lr_mini=args.lr_mini,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_epochs=args.num_train_epochs,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
    )

    training_args = GeneralConfig(
        num_train_epochs=args.num_train_epochs,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )

    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    output_dir = f"{args.output_dir}/{now_time}/"
    os.makedirs(output_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_computer=loss_computer,
        metrics_computer=metrics_computer,
        training_args=training_args,
        output_dir=output_dir,
    )

    trainer.train()


def main():
    os.chdir(sys.path[0])
    args = get_args()
    train(args)

if __name__ == "__main__":
    main()






    

    








