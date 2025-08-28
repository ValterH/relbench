import os
import argparse

import pandas as pd
import torch
from torch_geometric.seed import seed_everything

from relbench.base import Dataset, TaskType, Database
from relbench.datasets import get_dataset
from relbench.base.task_synthetic import SyntheticTask

def get_entity_tables(db: Database) -> list[str]:
    entity_tables = set()
    for table_name, table in db.table_dict.items():
        if table.time_col is None:
            continue
        for _, parent_table_name in table.fkey_col_to_pkey_table.items():
            # print(f"Adding {parent_table_name} as entity table (parent of {table_name})")
            entity_tables.add(parent_table_name)
    return list(entity_tables)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-amazon")
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-tasks", type=int, default=20) # 10 reg, 10 clf
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
entity_tables = get_entity_tables(dataset.get_db())

for i in range(0, args.num_tasks):
    if i >= args.num_tasks / 2:
        task_type = TaskType.REGRESSION
        task_name = f"synthetic_regression-{i - int(args.num_tasks / 2)}"
    else:
        task_type = TaskType.BINARY_CLASSIFICATION
        task_name = f"synthetic_classification-{i}"
    
    entity_table = entity_tables[i % len(entity_tables)]

    task = SyntheticTask(
        dataset=dataset,
        task_type=task_type,
        entity_table=entity_table,
        num_layers=args.num_layers,
        channels=args.channels,
        num_neighbors=args.num_neighbors,
        aggr=args.aggr,
        norm="batch_norm",
        timedelta=pd.Timedelta(days=30),
        temporal_strategy=args.temporal_strategy,
        device=device,
        db_cache_dir=os.path.expanduser(f"~/scratch/synthetic_tasks/{args.dataset}"),
    )

    table = task.get_table("train", mask_input_cols=False)
    task_dir = os.path.expanduser(f"~/scratch/relbench/{args.dataset}/tasks/{task_name}")
    os.makedirs(task_dir, exist_ok=True)
    table.save(f"{task_dir}/train.parquet")

