import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from model import Model
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument(
    "--tasks",
    type=list,
    default=[
        "driver-best-position",
        "driver-worst-position",
        "driver-position-year",
    ],  # , "driver-dnf"
)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--checkpoint-dir", type=str, default="ckpt")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
tasks: list[EntityTask] = [
    get_task(args.dataset, task, download=False) for task in args.tasks
]


stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

task_dict = {}
for task in tasks:
    task_name = str(task)
    task_dict[task_name] = {}
    task_dict[task_name]["clamp_min"] = None
    task_dict[task_name]["clamp_max"] = None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        task_dict[task_name]["out_channels"] = 1
        task_dict[task_name]["loss_fn"] = BCEWithLogitsLoss()
        task_dict[task_name]["tune_metric"] = "roc_auc"
        task_dict[task_name]["higher_is_better"] = True
    elif task.task_type == TaskType.REGRESSION:
        task_dict[task_name]["out_channels"] = 1
        task_dict[task_name]["loss_fn"] = L1Loss()
        task_dict[task_name]["tune_metric"] = "mae"
        task_dict[task_name]["higher_is_better"] = False
        # Get the clamp value at inference time
        train_table = task.get_table("train")
        task_dict[task_name]["clamp_min"], task_dict[task_name]["clamp_max"] = (
            np.percentile(train_table.df[task.target_col].to_numpy(), [2, 98])
        )
    # elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    #     out_channels = task.num_labels
    #     loss_fn = BCEWithLogitsLoss()
    #     tune_metric = "multilabel_auprc_macro"
    #     higher_is_better = True
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")

    dataloaders: Dict[str, NeighborLoader] = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        entity_table = table_input.nodes[0]
        dataloaders[split] = NeighborLoader(
            data,
            num_neighbors=[
                int(args.num_neighbors / 2**i) for i in range(args.num_layers)
            ],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=split == "train",
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

    task_dict[task_name]["dataloaders"] = dataloaders


def train() -> float:
    model.train()
    task_losses = dict()
    for task in np.random.permutation(tasks):
        task_name = str(task)
        task_info = task_dict[task_name]
        model.head = model_heads[task_name]
        loss_fn = task_info["loss_fn"]

        loader_dict = task_dict[task_name]["dataloaders"]
        loss_accum = count_accum = 0
        steps = 0
        total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)

        pbar = tqdm(loader_dict["train"], total=total_steps)
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(
                batch,
                task.entity_table,
            )
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            loss = loss_fn(pred.float(), batch[entity_table].y.float())
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)

            steps += 1
            if steps > args.max_steps_per_epoch:
                break

            pbar.set_description(f"Task: {task_name} Loss: {loss_accum / count_accum}")
        task_losses[task] = loss_accum / count_accum

    return task_losses


@torch.no_grad()
def test(mode="val") -> np.ndarray:
    model.eval()

    task_results = dict()
    for task in tasks:
        task_name = str(task)
        task_info = task_dict[task_name]
        model.head = model_heads[task_name]
        pred_list = []
        loader = task_dict[task_name]["dataloaders"][mode]
        for batch in tqdm(loader):
            batch = batch.to(device)
            pred = model(
                batch,
                task.entity_table,
            )
            if task.task_type == TaskType.REGRESSION:
                clamp_min = task_info["clamp_min"]
                clamp_max = task_info["clamp_max"]
                assert clamp_min is not None
                assert clamp_max is not None
                pred = torch.clamp(pred, clamp_min, clamp_max)

            if task.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            ]:
                pred = torch.sigmoid(pred)

            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
        task_results[task_name] = torch.cat(pred_list, dim=0).numpy()
    return task_results


model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,  # Not used in multi-task setting NOTE: should be in line with downstream task
    aggr=args.aggr,
    norm="batch_norm",
).to(device)

model_head = model.head

model_heads = nn.ModuleDict()
for task in tasks:
    task_name = str(task)
    out_channels = task_dict[task_name]["out_channels"]
    model_heads[task_name] = torch.nn.Linear(args.channels, out_channels).to(device)

optimizer = torch.optim.Adam(
    [{"params": model.parameters()}, {"params": model_heads.parameters()}], lr=args.lr
)
state_dict = None

for epoch in range(1, args.epochs + 1):
    train_loss_dict = train()
    val_pred = test(mode="val")
    val_metrics_dict = {}
    for task in tasks:
        task_name = str(task)
        train_loss = train_loss_dict[task]
        val_metrics = task.evaluate(val_pred[task_name], task.get_table("val"))
        val_metrics_dict[task] = val_metrics
        print(
            f"Epoch: {epoch:02d}, Task: {task_name} Train loss: {train_loss}, Val metrics: {val_metrics}"
        )
        # restore the model head
        model.head = model_head
        # always take the last model state dict
        state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(mode="val")

test_pred = test(mode="test")

for task in tasks:
    task_name = str(task)

    val_metrics = task.evaluate(val_pred[task_name], task.get_table("val"))
    print(f"Best Val metrics for {task_name}: {val_metrics}")

    test_metrics = task.evaluate(test_pred[task_name])
    print(f"Best test metrics for {task_name}: {test_metrics}")

os.makedirs(os.path.join(args.checkpoint_dir, args.dataset), exist_ok=True)
# restore the model head
model.head = model_head
torch.save(
    model.state_dict(),
    os.path.join(args.checkpoint_dir, args.dataset, "model_multi2.pth"),
)
