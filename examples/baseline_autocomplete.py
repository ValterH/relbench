import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from torch_geometric.seed import seed_everything

from relbench.base import Dataset, EntityTask, Table, TaskType
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="predict-column")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--download",
    action="store_true",
    default=False,
    help="Download the dataset if not already present.",
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(args.seed)

task: EntityTask = get_task(args.dataset, args.task, download=args.download)
dataset: Dataset = task.dataset


train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")


def evaluate(train_table: Table, pred_table: Table, name: str) -> Dict[str, float]:
    is_test = task.target_col not in pred_table.df
    if name == "global_zero":
        pred = np.zeros(len(pred_table))
    elif name == "global_mean":
        mean = train_table.df[task.target_col].astype(float).values.mean()
        pred = np.ones(len(pred_table)) * mean
    elif name == "global_median":
        median = np.median(train_table.df[task.target_col].astype(float).values)
        pred = np.ones(len(pred_table)) * median
    elif name == "entity_mean":
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: "mean"})
        df.rename(columns={task.target_col: "__target__"}, inplace=True)
        df = pred_table.df.merge(df, how="left", on=fkey)
        pred = df["__target__"].fillna(0).astype(float).values
    elif name == "entity_median":
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: "median"})
        df.rename(columns={task.target_col: "__target__"}, inplace=True)
        df = pred_table.df.merge(df, how="left", on=fkey)
        pred = df["__target__"].fillna(0).astype(float).values
    elif name == "random":
        pred = np.random.rand(len(pred_table))
    elif name == "majority":
        past_target = train_table.df[task.target_col].astype(int)
        majority_label = int(past_target.mode().iloc[0])
        pred = torch.full((len(pred_table),), fill_value=majority_label)
    elif name == "majority_multilabel":
        past_target = train_table.df[task.target_col]
        majority = mode(np.stack(past_target.values), axis=0).mode[0]
        pred = np.stack([majority] * len(pred_table.df))
    elif name == "random_multilabel":
        num_labels = train_table.df[task.target_col].values[0].shape[0]
        pred = np.random.rand(len(pred_table), num_labels)
    else:
        raise ValueError("Unknown eval name called {name}.")
    return task.evaluate(pred, None if is_test else pred_table)


trainval_table_df = pd.concat([train_table.df, val_table.df], axis=0)
trainval_table = Table(
    df=trainval_table_df,
    fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
    pkey_col=train_table.pkey_col,
    time_col=train_table.time_col,
)

if task.task_type == TaskType.REGRESSION:
    eval_name_list = [
        "global_zero",
        "global_mean",
        "global_median",
        "entity_mean",
        "entity_median",
    ]

    for name in eval_name_list:
        train_metrics = evaluate(train_table, train_table, name=name)
        val_metrics = evaluate(train_table, val_table, name=name)
        test_metrics = evaluate(trainval_table, test_table, name=name)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")


elif task.task_type == TaskType.BINARY_CLASSIFICATION:
    eval_name_list = ["random", "majority"]
    for name in eval_name_list:
        train_metrics = evaluate(train_table, train_table, name=name)
        val_metrics = evaluate(train_table, val_table, name=name)
        test_metrics = evaluate(trainval_table, test_table, name=name)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")


elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    eval_name_list = ["random_multilabel", "majority_multilabel"]
    for name in eval_name_list:
        train_metrics = evaluate(train_table, train_table, name=name)
        val_metrics = evaluate(train_table, val_table, name=name)
        test_metrics = evaluate(trainval_table, test_table, name=name)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")

elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
    # NOTE: Only keep accuracy for multiclass classification (no probabilities)
    task.metrics = task.metrics[:1]
    eval_name_list = ["random", "majority"]
    for name in eval_name_list:
        train_metrics = evaluate(train_table, train_table, name=name)
        val_metrics = evaluate(train_table, val_table, name=name)
        test_metrics = evaluate(trainval_table, test_table, name=name)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")
